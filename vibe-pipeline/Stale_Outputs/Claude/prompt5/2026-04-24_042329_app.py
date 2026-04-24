import os
import uuid
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, abort

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg',
    'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'csv', 'zip', 'mp4', 'mp3'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_filename(filename):
    filename = os.path.basename(filename)
    filename = filename.replace(' ', '_')
    return filename

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>File Upload App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
        h1 { color: #333; }
        .upload-box { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .file-list { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
        input[type=file] { margin: 10px 0; display: block; }
        input[type=submit] { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        input[type=submit]:hover { background: #45a049; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        a { color: #4CAF50; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .flash { padding: 10px; background: #dff0d8; border: 1px solid #d6e9c6; color: #3c763d; border-radius: 5px; margin-bottom: 15px; }
        .error { background: #f2dede; border-color: #ebccd1; color: #a94442; }
        .empty { color: #999; font-style: italic; }
    </style>
</head>
<body>
    <h1>📁 File Upload App</h1>

    {% if message %}
    <div class="flash {{ 'error' if error else '' }}">{{ message }}</div>
    {% endif %}

    <div class="upload-box">
        <h2>Upload a File</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Upload File">
        </form>
    </div>

    <div class="file-list">
        <h2>Uploaded Files</h2>
        {% if files %}
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Filename</th>
                    <th>Size</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
            {% for f in files %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ f.name }}</td>
                <td>{{ f.size }}</td>
                <td>
                    <a href="/view/{{ f.stored_name }}" target="_blank">View</a> |
                    <a href="/download/{{ f.stored_name }}">Download</a>
                </td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="empty">No files uploaded yet.</p>
        {% endif %}
    </div>
</body>
</html>
"""

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def get_uploaded_files():
    files = []
    if not os.path.exists(UPLOAD_FOLDER):
        return files
    for fname in os.listdir(UPLOAD_FOLDER):
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            # Try to recover original name from stored name pattern uuid_originalname
            parts = fname.split('_', 1)
            display_name = parts[1] if len(parts) == 2 else fname
            files.append({
                'name': display_name,
                'stored_name': fname,
                'size': format_size(size),
                'mtime': os.path.getmtime(fpath)
            })
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return files

@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', '')
    error = request.args.get('error', False)
    files = get_uploaded_files()
    return render_template_string(INDEX_TEMPLATE, files=files, message=message, error=error)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file part in request.', error=True))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index', message='No file selected.', error=True))
    
    if not allowed_file(file.filename):
        return redirect(url_for('index', message=f'File type not allowed. Allowed types: {", ".join(sorted(ALLOWED_EXTENSIONS))}', error=True))
    
    original_name = safe_filename(file.filename)
    unique_id = uuid.uuid4().hex[:8]
    stored_name = f"{unique_id}_{original_name}"
    
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(save_path)
    
    return redirect(url_for('index', message=f'File "{original_name}" uploaded successfully!'))

@app.route('/view/<filename>')
def view_file(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name)

@app.route('/download/<filename>')
def download_file(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.exists(file_path):
        abort(404)
    parts = safe_name.split('_', 1)
    download_name = parts[1] if len(parts) == 2 else safe_name
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name, as_attachment=True, download_name=download_name)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)