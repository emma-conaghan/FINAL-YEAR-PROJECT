from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'mp4', 'mp3', 'zip', 'svg', 'webp'}

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload App</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
            color: #333;
            min-height: 100vh;
        }
        nav {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        nav a {
            color: white;
            text-decoration: none;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 6px;
            transition: background 0.3s;
        }
        nav a:hover {
            background: rgba(255,255,255,0.2);
        }
        nav .brand {
            font-size: 22px;
            font-weight: bold;
        }
        .nav-links {
            display: flex;
            gap: 10px;
        }
        .container {
            max-width: 900px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        h1 {
            margin-bottom: 20px;
            color: #444;
        }
        h2 {
            margin-bottom: 15px;
            color: #555;
        }
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            transition: all 0.3s;
            cursor: pointer;
            position: relative;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f1ff;
        }
        .upload-area.dragover {
            border-color: #764ba2;
            background: #e8e9ff;
        }
        .upload-area input[type="file"] {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            opacity: 0;
            cursor: pointer;
        }
        .upload-area p {
            font-size: 18px;
            color: #667eea;
            margin-bottom: 10px;
        }
        .upload-area span {
            font-size: 14px;
            color: #999;
        }
        .btn {
            display: inline-block;
            padding: 10px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.3s;
            font-weight: 500;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .btn-danger {
            background: #e74c3c;
            color: white;
            padding: 6px 14px;
            font-size: 13px;
        }
        .btn-danger:hover {
            background: #c0392b;
        }
        .btn-download {
            background: #27ae60;
            color: white;
            padding: 6px 14px;
            font-size: 13px;
        }
        .btn-download:hover {
            background: #219a52;
        }
        .btn-view {
            background: #3498db;
            color: white;
            padding: 6px 14px;
            font-size: 13px;
        }
        .btn-view:hover {
            background: #2980b9;
        }
        .file-list {
            list-style: none;
        }
        .file-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 18px;
            border: 1px solid #eee;
            border-radius: 8px;
            margin-bottom: 8px;
            transition: all 0.2s;
            background: #fafafa;
        }
        .file-item:hover {
            background: #f0f2ff;
            border-color: #ddd;
        }
        .file-info {
            display: flex;
            align-items: center;
            gap: 12px;
            flex: 1;
            min-width: 0;
        }
        .file-icon {
            font-size: 28px;
            width: 40px;
            text-align: center;
            flex-shrink: 0;
        }
        .file-name {
            font-weight: 500;
            color: #333;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .file-size {
            color: #999;
            font-size: 13px;
        }
        .file-actions {
            display: flex;
            gap: 6px;
            flex-shrink: 0;
        }
        .flash-message {
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-weight: 500;
        }
        .flash-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .flash-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        .empty-state p {
            font-size: 18px;
            margin-bottom: 10px;
        }
        .file-count {
            color: #999;
            font-size: 14px;
            margin-bottom: 15px;
        }
        #selected-file-name {
            margin-top: 10px;
            font-size: 14px;
            color: #667eea;
            font-weight: 500;
        }
        .preview-container {
            text-align: center;
            margin: 20px 0;
        }
        .preview-container img {
            max-width: 100%;
            max-height: 600px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .preview-container iframe {
            width: 100%;
            height: 600px;
            border: 1px solid #eee;
            border-radius: 8px;
        }
        .preview-container video, .preview-container audio {
            max-width: 100%;
            border-radius: 8px;
        }
        .back-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #667eea;
            text-decoration: none;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .back-link:hover {
            text-decoration: underline;
        }
        @media (max-width: 600px) {
            .file-item {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
            .file-actions {
                width: 100%;
                justify-content: flex-end;
            }
        }
    </style>
</head>
<body>
    <nav>
        <a href="/" class="brand">📁 FileVault</a>
        <div class="nav-links">
            <a href="/">Upload</a>
            <a href="/files">My Files</a>
        </div>
    </nav>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="flash-message flash-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
    {% block scripts %}{% endblock %}
</body>
</html>
"""

UPLOAD_TEMPLATE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>📤 Upload Files</h1>
    <p style="color: #777; margin-bottom: 20px;">Upload documents, images, and other files (max 50MB)</p>
    <form method="POST" action="/upload" enctype="multipart/form-data" id="upload-form">
        <div class="upload-area" id="drop-area">
            <input type="file" name="file" id="file-input" required>
            <p>📎 Drop your file here or click to browse</p>
            <span>Supported: images, documents, archives, media files</span>
            <div id="selected-file-name"></div>
        </div>
        <br>
        <button type="submit" class="btn btn-primary">⬆️ Upload File</button>
    </form>
</div>
{% endblock %}
{% block scripts %}
<script>
    const fileInput = document.getElementById('file-input');
    const dropArea = document.getElementById('drop-area');
    const selectedName = document.getElementById('selected-file-name');
    
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            selectedName.textContent = '✅ Selected: ' + this.files[0].name;
        }
    });
    
    dropArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });
    dropArea.addEventListener('dragleave', function(e) {
        this.classList.remove('dragover');
    });
    dropArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        fileInput.files = e.dataTransfer.files;
        if (fileInput.files.length > 0) {
            selectedName.textContent = '✅ Selected: ' + fileInput.files[0].name;
        }
    });
</script>
{% endblock %}
"""

FILES_TEMPLATE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>📂 My Files</h1>
    {% if files %}
    <p class="file-count">{{ files|length }} file{{ 's' if files|length != 1 else '' }} uploaded</p>
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
                <a href="/view/{{ file.name }}" class="btn btn-view">👁️ View</a>
                <a href="/download/{{ file.name }}" class="btn btn-download">⬇️ Download</a>
                <form method="POST" action="/delete/{{ file.name }}" style="display:inline;" onsubmit="return confirm('Delete this file?');">
                    <button type="submit" class="btn btn-danger">🗑️ Delete</button>
                </form>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <p>📭 No files uploaded yet</p>
        <span>Go to <a href="/" style="color: #667eea;">Upload</a> to add files</span>
    </div>
    {% endif %}
</div>
{% endblock %}
"""

VIEW_TEMPLATE = """
{% extends "base" %}
{% block content %}
<a href="/files" class="back-link">← Back to files</a>
<div class="card">
    <h2>{{ filename }}</h2>
    <div class="preview-container">
        {% if file_type == 'image' %}
            <img src="/uploads/{{ filename }}" alt="{{ filename }}">
        {% elif file_type == 'pdf' %}
            <iframe src="/uploads/{{ filename }}"></iframe>
        {% elif file_type == 'video' %}
            <video controls>
                <source src="/uploads/{{ filename }}">
                Your browser does not support the video tag.
            </video>
        {% elif file_type == 'audio' %}
            <audio controls>
                <source src="/uploads/{{ filename }}">
                Your browser does not support the audio tag.
            </audio>
        {% elif file_type == 'text' %}
            <pre style="text-align:left; background:#f5f5f5; padding:20px; border-radius:8px; overflow-x:auto; white-space:pre-wrap; word-wrap:break-word;">{{ text_content }}</pre>
        {% else %}
            <div class="empty-state">
                <p>👀 Preview not available for this file type</p>
                <br>
                <a href="/download/{{ filename }}" class="btn btn-download">⬇️ Download instead</a>
            </div>
        {% endif %}
    </div>
    <div style="margin-top: 20px; text-align: center;">
        <a href