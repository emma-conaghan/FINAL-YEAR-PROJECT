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
        }
        nav a {
            color: #ecf0f1;
            text-decoration: none;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 5px;
            transition: background 0.3s;
        }
        nav a:hover { background: #34495e; }
        nav .brand { font-size: 20px; font-weight: bold; color: #3498db; }
        .container {
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
        }
        h1 { margin-bottom: 25px; color: #2c3e50; }
        .upload-area {
            background: white;
            border: 2px dashed #bdc3c7;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: border-color 0.3s;
        }
        .upload-area:hover { border-color: #3498db; }
        .upload-area input[type="file"] {
            margin: 15px 0;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .file-list { list-style: none; }
        .file-item {
            background: white;
            margin-bottom: 10px;
            padding: 15px 20px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: box-shadow 0.3s;
        }
        .file-item:hover { box-shadow: 0 3px 10px rgba(0,0,0,0.15); }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 28px; }
        .file-name {
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .file-size { color: #7f8c8d; font-size: 13px; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .file-actions a, .file-actions button {
            padding: 8px 14px;
            border-radius: 4px;
            font-size: 13px;
            text-decoration: none;
            border: none;
            cursor: pointer;
            color: white;
        }
        .btn-view { background: #3498db; }
        .btn-view:hover { background: #2980b9; }
        .btn-download { background: #27ae60; }
        .btn-download:hover { background: #219a52; }
        .btn-delete { background: #e74c3c; }
        .btn-delete:hover { background: #c0392b; }
        .message {
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .message-success { background: #d5f5e3; color: #1e8449; }
        .message-error { background: #fadbd8; color: #c0392b; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #95a5a6;
        }
        .empty-state p { font-size: 18px; margin-bottom: 20px; }
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
'''

UPLOAD_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Upload Files</h1>
{% if message %}
<div class="message {{ message_class }}">{{ message }}</div>
{% endif %}
<div class="upload-area">
    <p style="font-size: 40px; margin-bottom: 10px;">📤</p>
    <p style="font-size: 18px; color: #7f8c8d; margin-bottom: 20px;">Select files to upload</p>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="files" multiple><br><br>
        <button type="submit" class="btn">Upload Files</button>
    </form>
</div>
<p style="text-align:center; color: #95a5a6;">Maximum file size: 50 MB</p>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Uploaded Files ({{ files|length }})</h1>
{% if files %}
<ul class="file-list">
    {% for file in files %}
    <li class="file-item">
        <div class="file-info">
            <span class="file-icon">{{ file.icon }}</span>
            <div>
                <div class="file-name" title="{{ file.name }}">{{ file.name }}</div>
                <div class="file-size">{{ file.size }}</div>
            </div>
        </div>
        <div class="file-actions">
            <a href="/view/{{ file.name }}" class="btn-view" target="_blank">View</a>
            <a href="/download/{{ file.name }}" class="btn-download">Download</a>
            <form action="/delete/{{ file.name }}" method="post" style="display:inline;" onsubmit="return confirm('Delete this file?');">
                <button type="submit" class="btn-delete">Delete</button>
            </form>
        </div>
    </li>
    {% endfor %}
</ul>
{% else %}
<div class="empty-state">
    <p style="font-size: 60px;">📂</p>
    <p>No files uploaded yet</p>
    <a href="/" class="btn">Upload your first file</a>
</div>
{% endif %}
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📕', 'doc': '📘', 'docx': '📘', 'txt': '📄', 'rtf': '📄',
        'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'ppt': '📙', 'pptx': '📙',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵',
        'zip': '🗜️', 'rar': '🗜️', 'tar': '🗜️', 'gz': '🗜️',
        'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨',
        'json': '📋', 'xml': '📋',
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


def render_with_base(template_str, **kwargs):
    from jinja2 import Environment, BaseLoader, DictLoader
    templates = {
        'base': BASE_TEMPLATE,
        'page': template_str,
    }
    env = Environment(loader=DictLoader(templates))
    template = env.get_template('page')
    return template.render(**kwargs)


@app.route('/')
def index():
    message = request.args.get('message', '')
    message_class = request.args.get('message_class', '')
    return render_with_base(UPLOAD_TEMPLATE, message=message, message_class=message_class)


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return redirect(url_for('index', message='No file selected', message_class='message-error'))

    files = request.files.getlist('files')
    uploaded_count = 0

    for file in files:
        if file and file.filename and file.filename.strip():
            filename = file.filename.strip()
            # Simple sanitization
            filename = filename.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_count += 1

    if uploaded_count == 0:
        return redirect(url_for('index', message='No valid files were uploaded', message_class='message-error'))

    return redirect(url_for('index',
                            message=f'Successfully uploaded {uploaded_count} file(s)!',
                            message_class='message-success'))


@app.route('/files')
def file_list():
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
    return render_with_base(FILES_TEMPLATE, files=files)


@app.route('/view/<path:filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/delete/<path:filename>', methods=['POST'])
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        os.remove(filepath)
    return redirect(url_for('file_list'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)