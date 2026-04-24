from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'svg', 'webp', 'mp4', 'mp3', 'zip', 'html', 'css', 'js', 'json', 'xml'}

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload App</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; min-height: 100vh; }
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
        .navbar h1 { font-size: 1.4em; }
        .navbar a { color: #ecf0f1; text-decoration: none; margin-left: 20px; font-weight: 500; transition: color 0.2s; }
        .navbar a:hover { color: #3498db; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; padding: 30px; margin-bottom: 25px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 10px; padding: 40px; text-align: center; transition: border-color 0.3s, background 0.3s; cursor: pointer; }
        .upload-area:hover { border-color: #3498db; background: #f7fbff; }
        .upload-area.dragover { border-color: #2ecc71; background: #f0fff4; }
        .upload-area input[type="file"] { display: none; }
        .upload-area label { cursor: pointer; display: block; }
        .upload-area .icon { font-size: 48px; margin-bottom: 10px; }
        .upload-area p { color: #7f8c8d; margin-top: 8px; }
        .btn { display: inline-block; padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.95em; font-weight: 600; text-decoration: none; transition: background 0.2s, transform 0.1s; }
        .btn:active { transform: scale(0.97); }
        .btn-primary { background: #3498db; color: white; }
        .btn-primary:hover { background: #2980b9; }
        .btn-success { background: #2ecc71; color: white; }
        .btn-success:hover { background: #27ae60; }
        .btn-danger { background: #e74c3c; color: white; font-size: 0.8em; padding: 6px 14px; }
        .btn-danger:hover { background: #c0392b; }
        .btn-secondary { background: #95a5a6; color: white; font-size: 0.85em; padding: 8px 16px; }
        .btn-secondary:hover { background: #7f8c8d; }
        .file-list { list-style: none; }
        .file-list li { display: flex; align-items: center; justify-content: space-between; padding: 12px 15px; border-bottom: 1px solid #ecf0f1; transition: background 0.2s; }
        .file-list li:hover { background: #f8f9fa; }
        .file-list li:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 24px; flex-shrink: 0; }
        .file-name { font-weight: 500; color: #2c3e50; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .file-size { color: #95a5a6; font-size: 0.85em; flex-shrink: 0; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; margin-left: 15px; }
        .flash-messages { list-style: none; margin-bottom: 20px; }
        .flash-messages li { padding: 12px 18px; border-radius: 6px; margin-bottom: 8px; font-weight: 500; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .flash-info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .empty-state { text-align: center; padding: 40px; color: #95a5a6; }
        .empty-state .icon { font-size: 64px; margin-bottom: 15px; }
        .selected-file { margin-top: 15px; padding: 10px; background: #eaf6ff; border-radius: 6px; display: none; }
        #submit-btn { margin-top: 15px; display: none; }
        .file-count { background: #3498db; color: white; border-radius: 20px; padding: 2px 10px; font-size: 0.85em; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📁 File Upload App</h1>
        <nav>
            <a href="/">🏠 Home</a>
            <a href="/upload">⬆️ Upload</a>
            <a href="/files">📂 Files</a>
        </nav>
    </div>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
        <ul class="flash-messages">
            {% for category, message in messages %}
            <li class="flash-{{ category }}">{{ message }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

INDEX_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card" style="text-align: center; padding: 50px 30px;">
    <div style="font-size: 72px; margin-bottom: 20px;">📄</div>
    <h2 style="font-size: 1.8em; margin-bottom: 10px;">Welcome to File Upload App</h2>
    <p style="color: #7f8c8d; font-size: 1.1em; margin-bottom: 30px;">Upload, manage, and view your documents and images in one place.</p>
    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
        <a href="/upload" class="btn btn-primary" style="padding: 14px 32px; font-size: 1.05em;">⬆️ Upload Files</a>
        <a href="/files" class="btn btn-success" style="padding: 14px 32px; font-size: 1.05em;">📂 Browse Files <span class="file-count">{{ file_count }}</span></a>
    </div>
</div>
{% endblock %}
'''

UPLOAD_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>⬆️ Upload Files</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data" id="upload-form">
        <div class="upload-area" id="drop-zone">
            <label for="file-input">
                <div class="icon">📤</div>
                <h3>Drag & drop files here</h3>
                <p>or click to browse your computer</p>
                <p style="font-size: 0.85em; margin-top: 10px;">Supported: images, documents, archives (max 50MB)</p>
            </label>
            <input type="file" name="files" id="file-input" multiple>
        </div>
        <div class="selected-file" id="selected-files"></div>
        <button type="submit" class="btn btn-success" id="submit-btn">⬆️ Upload Files</button>
    </form>
</div>
<script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const selectedFiles = document.getElementById('selected-files');
    const submitBtn = document.getElementById('submit-btn');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        fileInput.files = e.dataTransfer.files;
        updateFileList();
    });

    fileInput.addEventListener('change', updateFileList);

    function updateFileList() {
        const files = fileInput.files;
        if (files.length > 0) {
            let html = '<strong>Selected files:</strong><ul style="margin-top:8px;margin-left:20px;">';
            for (let i = 0; i < files.length; i++) {
                const size = (files[i].size / 1024).toFixed(1);
                html += '<li>' + files[i].name + ' (' + size + ' KB)</li>';
            }
            html += '</ul>';
            selectedFiles.innerHTML = html;
            selectedFiles.style.display = 'block';
            submitBtn.style.display = 'inline-block';
        }
    }
</script>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>📂 Uploaded Files <span class="file-count">{{ files|length }}</span></h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li>
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <span class="file-name" title="{{ file.name }}">{{ file.name }}</span>
                <span class="file-size">{{ file.size }}</span>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="btn btn-primary" style="font-size:0.85em; padding:6px 14px;" target="_blank">👁️ View</a>
                <a href="/download/{{ file.name }}" class="btn btn-secondary" style="font-size:0.85em; padding:6px 14px;">⬇️ Download</a>
                <form method="POST" action="/delete/{{ file.name }}" style="display:inline;" onsubmit="return confirm('Delete {{ file.name }}?');">
                    <button type="submit" class="btn btn-danger">🗑️</button>
                </form>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <h3>No files uploaded yet</h3>
        <p style="margin-top: 10px;">Go to the <a href="/upload">upload page</a> to add files.</p>
    </div>
    {% endif %}
</div>
{% endblock %}
'''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📕', 'doc': '📘', 'docx': '📘', 'xls': '📗', 'xlsx': '📗',
        'csv': '📊', 'txt': '📝', 'html': '🌐', 'css': '🎨', 'js': '⚡',
        'json': '📋', 'xml': '📋', 'zip': '📦',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️',
        'svg': '🖼️', 'webp': '🖼️', 'mp4': '🎬', 'mp3': '🎵',
    }
    return icons.get(ext, '📄')


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_uploaded_files():
    files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                files.append({
                    'name': filename,
                    'size': format_size(os.path.getsize(filepath)),
                    'icon': get_file_icon(filename),
                })
    return files


from jinja2 import DictLoader

template_loader = DictLoader({
    'base': BASE_TEMPLATE,
    'index': INDEX_TEMPLATE,