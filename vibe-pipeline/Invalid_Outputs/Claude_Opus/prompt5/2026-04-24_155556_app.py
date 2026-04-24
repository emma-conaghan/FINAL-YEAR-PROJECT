from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'svg', 'webp', 'mp4', 'mp3', 'zip'}

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Manager</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; min-height: 100vh; }
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .navbar h1 { font-size: 1.5em; }
        .navbar a { color: white; text-decoration: none; margin-left: 20px; padding: 8px 16px; border-radius: 5px; transition: background 0.3s; }
        .navbar a:hover { background: rgba(255,255,255,0.1); }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); padding: 30px; margin-bottom: 20px; }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 10px; padding: 40px; text-align: center; transition: border-color 0.3s, background 0.3s; margin-bottom: 15px; }
        .upload-area:hover { border-color: #3498db; background: #f7fbff; }
        .upload-area input[type="file"] { margin: 10px 0; }
        .btn { display: inline-block; padding: 10px 24px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; text-decoration: none; transition: background 0.3s; }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .btn-danger { background: #e74c3c; font-size: 0.85em; padding: 6px 14px; }
        .btn-danger:hover { background: #c0392b; }
        .btn-sm { font-size: 0.85em; padding: 6px 14px; }
        .file-list { list-style: none; }
        .file-list li { display: flex; align-items: center; justify-content: space-between; padding: 12px 15px; border-bottom: 1px solid #ecf0f1; transition: background 0.2s; }
        .file-list li:hover { background: #f7f9fa; }
        .file-list li:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 1.5em; width: 40px; text-align: center; }
        .file-name { font-weight: 500; word-break: break-all; }
        .file-size { color: #7f8c8d; font-size: 0.85em; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .flash-msg { padding: 12px 20px; border-radius: 5px; margin-bottom: 15px; }
        .flash-success { background: #d5f5e3; color: #1e8449; }
        .flash-error { background: #fadbd8; color: #c0392b; }
        .empty-state { text-align: center; padding: 40px; color: #95a5a6; }
        .empty-state .icon { font-size: 3em; margin-bottom: 10px; }
        .preview-container { text-align: center; margin: 20px 0; }
        .preview-container img { max-width: 100%; max-height: 500px; border-radius: 5px; }
        .preview-container iframe { width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 5px; }
        .preview-container video, .preview-container audio { max-width: 100%; }
        .back-link { display: inline-block; margin-bottom: 15px; color: #3498db; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📁 File Manager</h1>
        <nav>
            <a href="/">Upload</a>
            <a href="/files">Files</a>
        </nav>
    </div>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="flash-msg flash-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

UPLOAD_PAGE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>📤 Upload Files</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-area">
            <p style="font-size:2em; margin-bottom:10px;">📎</p>
            <p>Select files to upload</p>
            <p style="color:#95a5a6; font-size:0.9em; margin-top:5px;">Max file size: 16MB</p>
            <input type="file" name="files" multiple>
        </div>
        <button type="submit" class="btn btn-success">Upload Files</button>
    </form>
</div>
<div class="card">
    <h2>📋 Recent Uploads</h2>
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
                <a href="/view/{{ file.name }}" class="btn btn-sm">View</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% if files|length > 5 %}
    <p style="text-align:center; margin-top:15px;"><a href="/files" class="btn">View All Files</a></p>
    {% endif %}
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet</p>
    </div>
    {% endif %}
</div>
{% endblock %}
"""

FILES_PAGE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>📂 All Uploaded Files ({{ files|length }})</h2>
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
                <a href="/view/{{ file.name }}" class="btn btn-sm">View</a>
                <a href="/download/{{ file.name }}" class="btn btn-sm btn-success">Download</a>
                <a href="/delete/{{ file.name }}" class="btn btn-sm btn-danger" onclick="return confirm('Delete this file?');">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet</p>
        <a href="/" class="btn" style="margin-top:15px;">Upload Files</a>
    </div>
    {% endif %}
</div>
{% endblock %}
"""

VIEW_PAGE = """
{% extends "base" %}
{% block content %}
<a href="/files" class="back-link">← Back to files</a>
<div class="card">
    <h2>{{ filename }}</h2>
    <div class="preview-container">
        {% if filetype == 'image' %}
            <img src="/uploads/{{ filename }}" alt="{{ filename }}">
        {% elif filetype == 'pdf' %}
            <iframe src="/uploads/{{ filename }}"></iframe>
        {% elif filetype == 'video' %}
            <video controls>
                <source src="/uploads/{{ filename }}">
                Your browser does not support the video tag.
            </video>
        {% elif filetype == 'audio' %}
            <audio controls>
                <source src="/uploads/{{ filename }}">
                Your browser does not support the audio tag.
            </audio>
        {% elif filetype == 'text' %}
            <div style="text-align:left; background:#f8f9fa; padding:20px; border-radius:5px; white-space:pre-wrap; font-family:monospace; max-height:500px; overflow:auto;">{{ text_content }}</div>
        {% else %}
            <div class="empty-state">
                <div class="icon">📄</div>
                <p>Preview not available for this file type</p>
            </div>
        {% endif %}
    </div>
    <div style="margin-top:20px; text-align:center;">
        <a href="/download/{{ filename }}" class="btn btn-success">Download File</a>
    </div>
</div>
{% endblock %}
"""


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📕', 'doc': '📘', 'docx': '📘',
        'xls': '📗', 'xlsx': '📗', 'csv': '📗',
        'txt': '📝', 'png': '🖼️', 'jpg': '🖼️',
        'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️',
        'svg': '🖼️', 'webp': '🖼️', 'mp4': '🎬',
        'mp3': '🎵', 'zip': '📦',
    }
    return icons.get(ext, '📄')


def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext in ('png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg', 'webp'):
        return 'image'
    elif ext == 'pdf':
        return 'pdf'
    elif ext == 'mp4':
        return 'video'
    elif ext == 'mp3':
        return 'audio'
    elif ext in ('txt', 'csv'):
        return 'text'
    return 'other'


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_files_list():
    files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                files.append({
                    'name': filename,
                    'size': format_size(os.path.getsize(filepath)),
                    'icon': get_file_icon(filename),
                })
    return files


# Custom Jinja2 loader to handle our inline templates
from jinja2 import BaseLoader, TemplateNotFound

class InlineLoader(BaseLoader):
    def __init__(self, templates):
        self.templates = templates

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)


templates_dict = {
    'base': BASE_TEMPLATE,
    'upload': UPLOAD_PAGE,
    'files': FILES_PAGE,
    'view': VIEW_PAGE,
}

app.jinja_loader = InlineLoader(templates_dict)


@app.route('/')
def index():
    files = get_files_list()
    return render_template_string(UPLOAD_PAGE, files=files)


# Override render to use our custom loader
from flask import render_template

@app.route('/', endpoint='index_page')
def index_page():
    files = get_files_list()
    return render_template('upload', files=files)


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index_page'))

    uploaded_files = request.files.getlist('