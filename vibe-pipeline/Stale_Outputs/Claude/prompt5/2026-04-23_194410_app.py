import anthropic
import base64
import os
import uuid
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import json
import mimetypes

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

client = anthropic.Anthropic()

def analyze_file_with_claude(file_path: Path, mime_type: str) -> str:
    """Analyze uploaded file using Claude API."""
    try:
        if mime_type.startswith('image/'):
            with open(file_path, 'rb') as f:
                file_data = base64.standard_b64encode(f.read()).decode('utf-8')
            
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": file_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Please provide a brief description of this image in 2-3 sentences."
                            }
                        ],
                    }
                ],
            )
            return message.content[0].text
        elif mime_type == 'application/pdf':
            with open(file_path, 'rb') as f:
                file_data = base64.standard_b64encode(f.read()).decode('utf-8')
            
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": file_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Please provide a brief summary of this document in 2-3 sentences."
                            }
                        ],
                    }
                ],
            )
            return message.content[0].text
        elif mime_type.startswith('text/'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(5000)
            
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please provide a brief summary of this text content in 2-3 sentences:\n\n{content}"
                    }
                ],
            )
            return message.content[0].text
        else:
            return f"File type {mime_type} - no analysis available"
    except Exception as e:
        return f"Analysis unavailable: {str(e)}"

def get_files_metadata():
    """Get metadata for all uploaded files."""
    metadata_file = UPLOAD_DIR / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def save_file_metadata(file_id: str, metadata: dict):
    """Save file metadata."""
    all_metadata = get_files_metadata()
    all_metadata[file_id] = metadata
    metadata_file = UPLOAD_DIR / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f)

def parse_multipart_form_data(body: bytes, boundary: str):
    """Parse multipart form data."""
    parts = {}
    boundary_bytes = f'--{boundary}'.encode()
    
    sections = body.split(boundary_bytes)
    
    for section in sections[1:]:
        if section.startswith(b'--'):
            continue
        
        if b'\r\n\r\n' in section:
            headers_part, content = section.split(b'\r\n\r\n', 1)
            
            if content.endswith(b'\r\n'):
                content = content[:-2]
            
            headers_str = headers_part.decode('utf-8', errors='ignore')
            
            filename = None
            field_name = None
            
            for line in headers_str.split('\r\n'):
                if 'Content-Disposition' in line:
                    if 'filename=' in line:
                        filename_start = line.index('filename=') + 9
                        filename = line[filename_start:].strip('"').strip()
                    if 'name=' in line:
                        name_start = line.index('name=') + 5
                        name_end = line.find(';', name_start)
                        if name_end == -1:
                            name_end = len(line)
                        field_name = line[name_start:name_end].strip('"').strip()
            
            if field_name:
                if filename:
                    parts[field_name] = {'filename': filename, 'content': content}
                else:
                    parts[field_name] = content.decode('utf-8', errors='ignore')
    
    return parts

UPLOAD_FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>File Upload Service</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .upload-form { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .file-list { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 20px; }
        .file-item { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .file-item:last-child { border-bottom: none; }
        .btn { background: #007bff; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn:hover { background: #0056b3; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #1e7e34; }
        input[type="file"] { margin: 10px 0; }
        .analysis { font-size: 0.85em; color: #666; margin-top: 5px; }
        .file-info { flex: 1; }
        .file-actions { display: flex; gap: 10px; }
        .loading { color: #999; font-style: italic; }
    </style>
</head>
<body>
    <h1>📁 File Upload Service</h1>
    
    <div class="upload-form">
        <h2>Upload a File</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*,.pdf,.txt,.doc,.docx" required><br>
            <button type="submit" class="btn btn-success">Upload File</button>
        </form>
    </div>
    
    <div class="file-list">
        <h2>Uploaded Files</h2>
        {file_list}
    </div>
</body>
</html>
"""

class FileUploadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index':
            self.serve_index()
        elif path.startswith('/files/'):
            file_id = path[7:]
            self.serve_file(file_id)
        elif path.startswith('/view/'):
            file_id = path[6:]
            self.serve_view(file_id)
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/upload':
            self.handle_upload()
        else:
            self.send_error(404)
    
    def serve_index(self):
        """Serve the main page with file list."""
        all_metadata = get_files_metadata()
        
        if not all_metadata:
            file_list_html = '<p>No files uploaded yet.</p>'
        else:
            file_items = []
            for file_id, metadata in sorted(all_metadata.items(), key=lambda x: x[1].get('upload_time', ''), reverse=True):
                analysis = metadata.get('analysis', '')
                analysis_html = f'<div class="analysis">🤖 {analysis}</div>' if analysis else ''
                
                file_items.append(f"""
                <div class="file-item">
                    <div class="file-info">
                        <strong>{metadata['original_name']}</strong>
                        <span style="color: #999; font-size: 0.85em; margin-left: 10px;">
                            {metadata.get('size', 0) // 1024} KB
                        </span>
                        {analysis_html}
                    </div>
                    <div class="file-actions">
                        <a href="/view/{file_id}" class="btn">👁 View</a>
                        <a href="/files/{file_id}" class="btn" download="{metadata['original_name']}">⬇ Download</a>
                    </div>
                </div>
                """)
            
            file_list_html = ''.join(file_items)
        
        html = UPLOAD_FORM_HTML.format(file_list=file_list_html)
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_file(self, file_id: str):
        """Serve a file for download."""
        all_metadata = get_files_metadata()
        
        if file_id not in all_metadata:
            self.send_error(404)
            return
        
        metadata = all_metadata[file_id]
        file_path = UPLOAD_DIR / metadata['stored_name']
        
        if not file_path.exists():
            self.send_error(404)
            return
        
        mime_type = metadata.get('mime_type', 'application/octet-stream')
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        self.send_response(200)
        self.send_header('Content-Type', mime_type)
        self.send_header('Content-Disposition', f'attachment; filename="{metadata["original_name"]}"')
        self.send_header('Content-Length', str(len(file_data)))
        self.end_headers()
        self.wfile.write(file_data)
    
    def serve_view(self, file_id: str):
        """Serve a file for viewing in browser."""
        all_metadata = get_files_metadata()
        
        if file_id not in all_metadata:
            self.send_error(404)
            return
        
        metadata = all_metadata[file_id]
        file_path = UPLOAD_DIR / metadata['stored_name']
        
        if not file_path.exists():
            self.send_error(404)
            return
        
        mime_type = metadata.get('mime_type', 'application/octet-stream')
        
        if mime_type.startswith('image/') or mime_type == 'application/pdf' or mime_type.startswith('text/'):
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', mime_type)
            self.send_header('Content-Disposition', f'inline; filename="{metadata["original_name"]}"')
            self.send_header('Content-Length', str(len(file_data)))
            self.end_headers()
            self.wfile.write(file_data)
        else:
            self.serve_file(file_id)
    
    def handle_upload(self):
        """Handle file upload."""
        content_type = self.headers.get('Content-Type', '')
        
        if 'multipart/form-data' not in content_type:
            self.send_error(400, 'Expected multipart/form-data')
            return
        
        boundary = content_type.split('boundary=')[1].strip()
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        parts = parse_multipart_form_data(body, boundary)
        
        if 'file' not in parts or not isinstance(parts['file'], dict):
            self.send_error(400, 'No file provided')
            return
        
        file_info = parts['file']
        original_filename = file_info['filename']
        file_content = file_info['content']
        
        if not original_filename:
            self.send_error(400, 'No filename provided')
            return
        
        file_id = str(uuid.uuid4())
        
        file_ext = Path(original_filename).suffix
        stored_filename = f"{file_id}{file_ext}"
        file_path = UPLOAD_DIR / stored_filename
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        mime_type, _ = mimetypes.guess_type(original_filename)
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        print(f"Analyzing {original_filename} with Claude...")
        analysis = analyze_file_with_claude(file_path, mime_type)
        
        import datetime
        metadata = {
            'original_name': original_filename,
            'stored_name': stored_filename,
            'mime_type': mime_type,
            'size': len(file_content),
            'analysis': analysis,
            'upload_time': datetime.datetime.now().isoformat()
        }
        
        save_file_metadata(file_id, metadata)
        
        self.send_response(302)
        self.send_header('Location', '/')
        self.end_headers()
    
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

def run_server(host='localhost', port=8080):
    """Run the HTTP server."""
    server = HTTPServer((host, port), FileUploadHandler)
    print(f"Starting file upload server at http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()

if __name__ == '__main__':
    run_server()