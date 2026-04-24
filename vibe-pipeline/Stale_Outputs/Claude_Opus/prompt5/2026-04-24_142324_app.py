from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload App</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; }
        .navbar { background: #2c3e50; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; }
        .navbar a { color: white; text-decoration: none; font-size: 16px; margin-left: 20px; }
        .navbar a:hover { text-decoration: underline; }
        .navbar .brand { font-size: 22px; font-weight: bold; color: #ecf0f1; }
        .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 30px; }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 10px; padding: 40px; text-align: center; margin-bottom: 20px; transition: border-color 0.3s; }
        .upload-area:hover { border-color: #3498db; }
        .upload-area input[type="file"] { margin: 10px 0; }
        .btn { display: inline-block; padding: 10px 25px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; text-decoration: none; }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .file-list { list-style: none; }
        .file-list li { display: flex; align-items: center; justify-content: space-between; padding: 12px 15px; border-bottom: 1px solid #ecf0f1; transition: background 0.2s; }
        .file-list li:hover { background: #f8f9fa; }
        .file-list li:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 24px; }
        .file-name { font-weight: 500; word-break: break-all; }
        .file-size { color: #7f8c8d; font-size: 13px; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .file-actions a { padding: 6px 14px; font-size: 13px; border-radius: 4px; text-decoration: none; color: white; }
        .file-actions .view-btn { background: #3498db; }
        .file-actions .view-btn:hover { background: #2980b9; }
        .file-actions .download-btn { background: #27ae60; }
        .file-actions .download-btn:hover { background: #219a52; }
        .file-actions .delete-btn { background: #e74c3c; }
        .file-actions .delete-btn:hover { background: #c0392b; }
        .message { padding: 12px 20px; border-radius: 5px; margin-bottom: 20px; }
        .message-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .empty-state { text-align: center; padding: 40px; color: #7f8c8d; }
        .empty-state .icon { font-size: 48px; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="navbar">
        <span class="brand">📁 File Upload App</span>
        <div>
            <a href="/">Home</a>
            <a href="/upload">Upload</a>
            <a href="/files">Files</a>
        </div>
    </div>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

INDEX_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card" style="text-align: center;">
    <h2>Welcome to File Upload App</h2>
    <p style="margin-bottom: 25px; color: #7f8c8d; font-size: 16px;">
        Upload documents and images, then view or download them anytime.
    </p>
    <div style="display: flex; gap: 15px; justify-content: center;">
        <a href="/upload" class="btn btn-success" style="font-size: 17px; padding: 12px 30px;">📤 Upload Files</a>
        <a href="/files" class="btn" style="font-size: 17px; padding: 12px 30px;">📂 Browse Files</a>
    </div>
</div>
<div class="card">
    <h2>Recent Uploads</h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files[:5] %}
        <li>
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <div>
                    <div class="file-name">{{ file.name }}</div>
                    <div class="file-size">{{ file.size }}</div>
                </div>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="view-btn">View</a>
                <a href="/download/{{ file.name }}" class="download-btn">Download</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% if files|length > 5 %}
    <div style="text-align: center; margin-top: 15px;">
        <a href="/files" class="btn">View All Files ({{ files|length }})</a>
    </div>
    {% endif %}
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet. Start by uploading some files!</p>
    </div>
    {% endif %}
</div>
{% endblock %}
'''

UPLOAD_TEMPLATE = '''
{% extends "base" %}
{% block content %}
{% if message %}
<div class="message {{ message_class }}">{{ message }}</div>
{% endif %}
<div class="card">
    <h2>📤 Upload Files</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-area" id="dropArea">
            <p style="font-size: 18px; margin-bottom: 10px;">Drag & drop files here or click to browse</p>
            <input type="file" name="files" multiple id="fileInput">
            <p style="color: #95a5a6; margin-top: 10px; font-size: 13px;">Max file size: 50MB</p>
        </div>
        <div style="text-align: center;">
            <button type="submit" class="btn btn-success" style="font-size: 16px; padding: 12px 35px;">Upload Files</button>
        </div>
    </form>
</div>
<script>
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    ['dragenter', 'dragover'].forEach(evt => {
        dropArea.addEventListener(evt, e => { e.preventDefault(); dropArea.style.borderColor = '#3498db'; dropArea.style.background = '#ebf5fb'; });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropArea.addEventListener(evt, e => { e.preventDefault(); dropArea.style.borderColor = '#bdc3c7'; dropArea.style.background = 'transparent'; });
    });
    dropArea.addEventListener('drop', e => { fileInput.files = e.dataTransfer.files; });
</script>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
{% if message %}
<div class="message {{ message_class }}">{{ message }}</div>
{% endif %}
<div class="card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h2>📂 All Files ({{ files|length }})</h2>
        <a href="/upload" class="btn btn-success">📤 Upload More</a>
    </div>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li>
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <div>
                    <div class="file-name">{{ file.name }}</div>
                    <div class="file-size">{{ file.size }}</div>
                </div>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="view-btn">View</a>
                <a href="/download/{{ file.name }}" class="download-btn">Download</a>
                <a href="/delete/{{ file.name }}" class="delete-btn" onclick="return confirm('Delete {{ file.name }}?');">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet.</p>
        <a href="/upload" class="btn" style="margin-top: 15px;">Upload Files</a>
    </div>
    {% endif %}
</div>
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📄', 'doc': '📝', 'docx': '📝', 'txt': '📃', 'rtf': '📃',
        'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'ppt': '📑', 'pptx': '📑',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵',
        'zip': '🗜️', 'rar': '🗜️', 'tar': '🗜️', 'gz': '🗜️', '7z': '🗜️',
        'py': '🐍', 'js': '⚡', 'html': '🌐', 'css': '🎨', 'json': '📋', 'xml': '📋',
    }
    return icons.get(ext, '📁')


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_files_list():
    files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                files.append({
                    'name': filename,
                    'size': format_size(size),
                    'icon': get_file_icon(filename),
                })
    return files


def render_with_base(template_str, **kwargs):
    from jinja2 import Environment, BaseLoader
    env = Environment(loader=BaseLoader())
    base = env.from_string(BASE_TEMPLATE)
    env.globals['base'] = base

    combined = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '')

    full_template = template_str.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', '')
    final = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', full_template)

    from jinja2 import Template
    t = Template(final)
    return t.render(**kwargs)


@app.route('/')
def index():
    files = get_files_list()
    return render_with_base(INDEX_TEMPLATE, files=files)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    message = None
    message_class = ''
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or all(f.filename == '' for f in uploaded_files):
            message = 'No files selected. Please choose at least one file.'
            message_class = 'message-error'
        else:
            count = 0
            for f in uploaded_files:
                if f.filename != '':
                    from werkzeug.utils import secure_filename
                    filename = secure_filename(f.filename)