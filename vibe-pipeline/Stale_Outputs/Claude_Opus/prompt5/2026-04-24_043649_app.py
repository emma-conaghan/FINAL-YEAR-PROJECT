from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

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
            background: #2c3e50;
            padding: 15px 30px;
            display: flex;
            align-items: center;
            gap: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        nav a {
            color: #ecf0f1;
            text-decoration: none;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 5px;
            transition: background 0.3s;
        }
        nav a:hover {
            background: #34495e;
        }
        nav .brand {
            font-size: 20px;
            font-weight: bold;
            color: #3498db;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        h1 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            margin-bottom: 20px;
        }
        .upload-area input[type="file"] {
            margin: 15px 0;
        }
        .btn {
            display: inline-block;
            padding: 10px 24px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            text-decoration: none;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #2980b9;
        }
        .btn-success {
            background: #27ae60;
        }
        .btn-success:hover {
            background: #219a52;
        }
        .btn-danger {
            background: #e74c3c;
        }
        .btn-danger:hover {
            background: #c0392b;
        }
        .file-list {
            list-style: none;
        }
        .file-list li {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
            transition: background 0.2s;
        }
        .file-list li:hover {
            background: #f8f9fa;
        }
        .file-list li:last-child {
            border-bottom: none;
        }
        .file-info {
            display: flex;
            align-items: center;
            gap: 12px;
            flex: 1;
        }
        .file-icon {
            font-size: 24px;
            width: 40px;
            text-align: center;
        }
        .file-name {
            font-weight: 500;
            word-break: break-all;
        }
        .file-size {
            color: #888;
            font-size: 13px;
        }
        .file-actions {
            display: flex;
            gap: 8px;
        }
        .file-actions a {
            padding: 6px 14px;
            font-size: 14px;
            border-radius: 4px;
            text-decoration: none;
            color: white;
        }
        .message {
            padding: 12px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .message-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .message-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #888;
        }
        .empty-state .icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <nav>
        <a href="/" class="brand">📁 FileVault</a>
        <a href="/">Upload</a>
        <a href="/files">My Files</a>
    </nav>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

UPLOAD_TEMPLATE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>Upload Files</h1>
    {% if message %}
    <div class="message {{ message_class }}">{{ message }}</div>
    {% endif %}
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-area">
            <div style="font-size: 48px; margin-bottom: 10px;">📤</div>
            <p style="margin-bottom: 15px; font-size: 18px; color: #555;">Select files to upload</p>
            <input type="file" name="files" multiple>
            <p style="margin-top: 10px; color: #888; font-size: 13px;">Maximum file size: 50MB</p>
        </div>
        <button type="submit" class="btn btn-success">⬆ Upload Files</button>
    </form>
</div>
{% endblock %}
"""

FILES_TEMPLATE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>Uploaded Files ({{ files|length }})</h1>
    {% if message %}
    <div class="message {{ message_class }}">{{ message }}</div>
    {% endif %}
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
                <a href="/view/{{ file.name }}" class="btn" target="_blank">👁 View</a>
                <a href="/download/{{ file.name }}" class="btn btn-success">⬇ Download</a>
                <a href="/delete/{{ file.name }}" class="btn btn-danger" onclick="return confirm('Delete {{ file.name }}?')">🗑</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet.</p>
        <a href="/" class="btn" style="margin-top: 15px; display: inline-block;">Upload your first file</a>
    </div>
    {% endif %}
</div>
{% endblock %}
"""


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icon_map = {
        'pdf': '📄',
        'doc': '📝', 'docx': '📝',
        'xls': '📊', 'xlsx': '📊',
        'ppt': '📑', 'pptx': '📑',
        'txt': '📃', 'csv': '📃', 'log': '📃',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵',
        'zip': '🗜️', 'rar': '🗜️', 'tar': '🗜️', 'gz': '🗜️', '7z': '🗜️',
        'py': '🐍', 'js': '⚡', 'html': '🌐', 'css': '🎨',
        'json': '📋', 'xml': '📋',
    }
    return icon_map.get(ext, '📁')


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_uploaded_files():
    files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                files.append({
                    'name': filename,
                    'size': format_size(size),
                    'icon': get_file_icon(filename),
                })
    return files


def render_with_base(template_string, **kwargs):
    from jinja2 import Environment, BaseLoader, DictLoader
    templates = {
        'base': BASE_TEMPLATE,
        'page': template_string,
    }
    env = Environment(loader=DictLoader(templates))
    template = env.get_template('page')
    return template.render(**kwargs)


@app.route('/')
def index():
    return render_with_base(UPLOAD_TEMPLATE, message=request.args.get('message'), message_class=request.args.get('message_class', ''))


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return redirect(url_for('index', message='No files selected.', message_class='message-error'))

    files = request.files.getlist('files')
    uploaded_count = 0

    for file in files:
        if file.filename and file.filename.strip():
            filename = file.filename.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_count += 1

    if uploaded_count == 0:
        return redirect(url_for('index', message='No files were uploaded. Please select at least one file.', message_class='message-error'))

    return redirect(url_for('files_list', message=f'Successfully uploaded {uploaded_count} file(s)!', message_class='message-success'))


@app.route('/files')
def files_list():
    files = get_uploaded_files()
    return render_with_base(FILES_TEMPLATE, files=files, message=request.args.get('message'), message_class=request.args.get('message_class', ''))


@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/delete/<filename>')
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        os.remove(filepath)
        return redirect(url_for('files_list', message=f'"{filename}" has been deleted.', message_class='message-success'))
    return redirect(url_for('files_list', message=f'File "{filename}" not found.', message_class='message-error'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)