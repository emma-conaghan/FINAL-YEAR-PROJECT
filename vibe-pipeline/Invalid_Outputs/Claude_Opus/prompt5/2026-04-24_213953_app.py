from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'svg', 'mp4', 'mp3', 'zip', 'html', 'css', 'js', 'json', 'xml'}

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
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .navbar h1 { font-size: 1.5em; }
        .navbar a { color: #ecf0f1; text-decoration: none; margin-left: 20px; padding: 8px 16px; border-radius: 4px; transition: background 0.3s; }
        .navbar a:hover { background: #34495e; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 20px; }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 8px; padding: 40px; text-align: center; transition: border-color 0.3s, background 0.3s; cursor: pointer; position: relative; }
        .upload-area:hover { border-color: #3498db; background: #f7fbff; }
        .upload-area input[type="file"] { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
        .upload-area p { font-size: 1.1em; color: #7f8c8d; margin-bottom: 10px; }
        .upload-area .icon { font-size: 3em; color: #bdc3c7; margin-bottom: 10px; }
        .btn { display: inline-block; padding: 10px 24px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; text-decoration: none; transition: background 0.3s; }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .file-list { list-style: none; }
        .file-item { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid #ecf0f1; transition: background 0.2s; }
        .file-item:hover { background: #f8f9fa; }
        .file-item:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 1.5em; width: 40px; text-align: center; flex-shrink: 0; }
        .file-name { font-weight: 500; word-break: break-all; }
        .file-size { color: #95a5a6; font-size: 0.85em; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .file-actions a { padding: 6px 14px; font-size: 0.9em; border-radius: 4px; text-decoration: none; color: white; }
        .file-actions .view-btn { background: #3498db; }
        .file-actions .view-btn:hover { background: #2980b9; }
        .file-actions .download-btn { background: #27ae60; }
        .file-actions .download-btn:hover { background: #219a52; }
        .file-actions .delete-btn { background: #e74c3c; }
        .file-actions .delete-btn:hover { background: #c0392b; }
        .flash-messages { margin-bottom: 20px; }
        .flash { padding: 12px 20px; border-radius: 4px; margin-bottom: 10px; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .empty-state { text-align: center; padding: 40px; color: #95a5a6; }
        .empty-state .icon { font-size: 4em; margin-bottom: 15px; }
        .file-count { color: #95a5a6; font-size: 0.9em; margin-left: 10px; }
        #fileName { margin-top: 10px; font-weight: 500; color: #2c3e50; }
        .preview-container { max-width: 100%; margin-top: 15px; text-align: center; }
        .preview-container img { max-width: 100%; max-height: 500px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📁 File Upload App</h1>
        <nav>
            <a href="/">Upload</a>
            <a href="/files">Files</a>
        </nav>
    </div>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
        <div class="flash-messages">
            {% for category, message in messages %}
            <div class="flash flash-{{ category }}">{{ message }}</div>
            {% endfor %}
        </div>
        {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

UPLOAD_PAGE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>Upload a File</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data" id="uploadForm">
        <div class="upload-area" id="dropArea">
            <div class="icon">⬆️</div>
            <p>Drag & drop a file here or click to browse</p>
            <p style="font-size: 0.85em; color: #bdc3c7;">Max file size: 50MB</p>
            <input type="file" name="file" id="fileInput" onchange="showFileName(this)">
            <div id="fileName"></div>
        </div>
        <br>
        <button type="submit" class="btn btn-success">Upload File</button>
    </form>
</div>
<script>
function showFileName(input) {
    var nameDiv = document.getElementById('fileName');
    if (input.files && input.files[0]) {
        nameDiv.textContent = 'Selected: ' + input.files[0].name + ' (' + formatBytes(input.files[0].size) + ')';
    }
}
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    var k = 1024;
    var sizes = ['Bytes', 'KB', 'MB', 'GB'];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
var dropArea = document.getElementById('dropArea');
['dragenter', 'dragover'].forEach(function(eventName) {
    dropArea.addEventListener(eventName, function(e) {
        e.preventDefault();
        dropArea.style.borderColor = '#3498db';
        dropArea.style.background = '#f7fbff';
    });
});
['dragleave', 'drop'].forEach(function(eventName) {
    dropArea.addEventListener(eventName, function(e) {
        e.preventDefault();
        dropArea.style.borderColor = '#bdc3c7';
        dropArea.style.background = '';
    });
});
dropArea.addEventListener('drop', function(e) {
    e.preventDefault();
    var fileInput = document.getElementById('fileInput');
    fileInput.files = e.dataTransfer.files;
    showFileName(fileInput);
});
</script>
{% endblock %}
'''

FILES_PAGE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>Uploaded Files <span class="file-count">({{ files|length }} file{{ 's' if files|length != 1 else '' }})</span></h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li class="file-item">
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <div>
                    <div class="file-name">{{ file.name }}</div>
                    <div class="file-size">{{ file.size }}</div>
                </div>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="view-btn" target="_blank">View</a>
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
        <br>
        <a href="/" class="btn">Upload a file</a>
    </div>
    {% endif %}
</div>
{% endblock %}
'''

VIEW_PAGE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>{{ filename }}</h2>
    <div class="preview-container">
        {% if filetype == 'image' %}
        <img src="/uploads/{{ filename }}" alt="{{ filename }}">
        {% elif filetype == 'pdf' %}
        <embed src="/uploads/{{ filename }}" type="application/pdf" width="100%" height="600px">
        {% elif filetype == 'text' %}
        <pre style="text-align: left; background: #f8f9fa; padding: 20px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;">{{ content }}</pre>
        {% elif filetype == 'video' %}
        <video controls style="max-width: 100%; max-height: 500px;">
            <source src="/uploads/{{ filename }}">
            Your browser does not support the video tag.
        </video>
        {% elif filetype == 'audio' %}
        <audio controls>
            <source src="/uploads/{{ filename }}">
            Your browser does not support the audio element.
        </audio>
        {% else %}
        <p>Preview not available for this file type.</p>
        {% endif %}
    </div>
    <br>
    <a href="/download/{{ filename }}" class="btn btn-success">Download</a>
    <a href="/files" class="btn">Back to Files</a>
</div>
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📄', 'doc': '📝', 'docx': '📝', 'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'txt': '📃', 'html': '🌐', 'css': '🎨', 'js': '⚡', 'json': '🔧', 'xml': '📋',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'webp': '🖼️', 'svg': '🖼️',
        'mp4': '🎬', 'mp3': '🎵', 'zip': '📦',
    }
    return icons.get(ext, '📁')


def format_size(size_bytes):
    if size_bytes == 0:
        return '0 Bytes'
    units = ['Bytes', 'KB', 'MB', 'GB']
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    return f"{size:.2f} {units[i]}"


def get_file_type(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext in ('png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg'):
        return 'image'
    elif ext == 'pdf':
        return 'pdf'
    elif ext in ('txt', 'html', 'css', 'js', 'json', 'xml', 'csv'):
        return 'text'
    elif ext in ('mp4',):
        return 'video'
    elif ext in ('mp3',):
        return 'audio'
    return 'other'


from jinja2 import DictLoader

template_loader = DictLoader({
    'base': BASE_TEMPLATE,
    'upload': UPLOAD_PAGE,
    'files': FILES_PAGE,
    'view': VIEW_PAGE,
})

app.jinja_loader = template_loader


@