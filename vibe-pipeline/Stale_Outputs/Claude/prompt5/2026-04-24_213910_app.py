import os
import uuid
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, abort

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg', 'webp',
    'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
    'txt', 'csv', 'zip', 'mp3', 'mp4', 'avi', 'mov'
}

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Manager</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
        h1, h2 { color: #333; }
        nav a { margin-right: 15px; color: #0077cc; text-decoration: none; font-weight: bold; }
        nav a:hover { text-decoration: underline; }
        .card { background: white; border-radius: 8px; padding: 20px; margin-top: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
        .upload-form input[type=file] { display: block; margin: 15px 0; }
        .upload-form button { background: #0077cc; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .upload-form button:hover { background: #005fa3; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #ddd; }
        th { background: #0077cc; color: white; }
        tr:hover { background: #f0f8ff; }
        a.file-link { color: #0077cc; text-decoration: none; }
        a.file-link:hover { text-decoration: underline; }
        .msg-success { color: green; font-weight: bold; margin-top: 10px; }
        .msg-error { color: red; font-weight: bold; margin-top: 10px; }
        .empty { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <h1>📁 File Manager</h1>
    <nav>
        <a href="{{ url_for('index') }}">Upload</a>
        <a href="{{ url_for('list_files') }}">My Files</a>
    </nav>
    {% block content %}{% endblock %}
</body>
</html>
"""

UPLOAD_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block content %}{% endblock %}",
    """
    <div class="card">
        <h2>Upload a File</h2>
        {% if message %}
            <p class="{{ 'msg-success' if success else 'msg-error' }}">{{ message }}</p>
        {% endif %}
        <form class="upload-form" method="POST" enctype="multipart/form-data" action="{{ url_for('upload_file') }}">
            <input type="file" name="file" required>
            <button type="submit">Upload</button>
        </form>
        <p style="color:#666; font-size:13px;">Allowed types: images, PDFs, Office documents, text files, zip, audio, video.</p>
    </div>
    """
)

LIST_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block content %}{% endblock %}",
    """
    <div class="card">
        <h2>Uploaded Files</h2>
        {% if files %}
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>File Name</th>
                    <th>Size</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
            {% for f in files %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td>{{ f.original_name }}</td>
                    <td>{{ f.size }}</td>
                    <td>
                        <a class="file-link" href="{{ url_for('view_file', filename=f.stored_name) }}" target="_blank">View/Open</a>
                        &nbsp;|&nbsp;
                        <a class="file-link" href="{{ url_for('download_file', filename=f.stored_name) }}">Download</a>
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
            <p class="empty">No files uploaded yet. <a href="{{ url_for('index') }}">Upload one now!</a></p>
        {% endif %}
    </div>
    """
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_human_readable_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.1f} GB"

def load_file_index():
    index_path = os.path.join(UPLOAD_FOLDER, '.index')
    files = []
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        files.append({'stored_name': parts[0], 'original_name': parts[1]})
    return files

def save_file_to_index(stored_name, original_name):
    index_path = os.path.join(UPLOAD_FOLDER, '.index')
    with open(index_path, 'a') as f:
        f.write(f"{stored_name}\t{original_name}\n")

@app.route('/', methods=['GET'])
def index():
    return render_template_string(UPLOAD_TEMPLATE, message=None, success=False)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template_string(UPLOAD_TEMPLATE, message="No file part in the request.", success=False)
    file = request.files['file']
    if file.filename == '':
        return render_template_string(UPLOAD_TEMPLATE, message="No file selected.", success=False)
    original_name = file.filename
    if not allowed_file(original_name):
        return render_template_string(UPLOAD_TEMPLATE, message=f"File type not allowed. Please upload a supported file type.", success=False)
    ext = original_name.rsplit('.', 1)[1].lower()
    stored_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, stored_name)
    file.save(save_path)
    save_file_to_index(stored_name, original_name)
    return render_template_string(UPLOAD_TEMPLATE, message=f"File '{original_name}' uploaded successfully!", success=True)

@app.route('/files', methods=['GET'])
def list_files():
    raw_files = load_file_index()
    files = []
    for entry in raw_files:
        file_path = os.path.join(UPLOAD_FOLDER, entry['stored_name'])
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            files.append({
                'stored_name': entry['stored_name'],
                'original_name': entry['original_name'],
                'size': get_human_readable_size(size_bytes)
            })
    return render_template_string(LIST_TEMPLATE, files=files)

@app.route('/view/<filename>')
def view_file(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, safe_name)

@app.route('/download/<filename>')
def download_file(filename):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    if not os.path.exists(file_path):
        abort(404)
    index = load_file_index()
    original_name = safe_name
    for entry in index:
        if entry['stored_name'] == safe_name:
            original_name = entry['original_name']
            break
    return send_from_directory(UPLOAD_FOLDER, safe_name, as_attachment=True, download_name=original_name)

if __name__ == '__main__':
    app.run(debug=True)