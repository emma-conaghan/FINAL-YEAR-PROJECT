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
            transition: background 0.2s;
        }
        nav a:hover { background: #34495e; }
        nav .brand { font-weight: bold; font-size: 20px; }
        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }
        h1 { margin-bottom: 25px; color: #2c3e50; }
        .upload-area {
            background: white;
            border: 2px dashed #bdc3c7;
            border-radius: 12px;
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
            padding: 12px 28px;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s;
            text-decoration: none;
            display: inline-block;
        }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .btn-danger { background: #e74c3c; font-size: 13px; padding: 6px 14px; }
        .btn-danger:hover { background: #c0392b; }
        .btn-small { font-size: 13px; padding: 6px 14px; }
        .file-list {
            list-style: none;
        }
        .file-item {
            background: white;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            transition: box-shadow 0.2s;
        }
        .file-item:hover { box-shadow: 0 3px 8px rgba(0,0,0,0.15); }
        .file-info {
            display: flex;
            align-items: center;
            gap: 12px;
            flex: 1;
            min-width: 0;
        }
        .file-icon { font-size: 28px; }
        .file-name {
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .file-size { color: #7f8c8d; font-size: 13px; }
        .file-actions {
            display: flex;
            gap: 8px;
            flex-shrink: 0;
        }
        .message {
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .message-success { background: #d5f4e6; color: #27ae60; }
        .message-error { background: #fde8e8; color: #e74c3c; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #95a5a6;
        }
        .empty-state .icon { font-size: 64px; margin-bottom: 15px; }
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
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <p style="font-size: 18px; color: #7f8c8d; margin-bottom: 10px;">Select files to upload</p>
        <input type="file" name="files" multiple>
        <br><br>
        <button type="submit" class="btn btn-success">Upload Files</button>
    </form>
</div>
<p style="color: #95a5a6; text-align: center;">Supports documents, images, and other file types up to 50MB</p>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Uploaded Files ({{ files|length }})</h1>
{% if message %}
<div class="message {{ message_class }}">{{ message }}</div>
{% endif %}
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
            <a href="/view/{{ file.name }}" class="btn btn-small" target="_blank">Open</a>
            <a href="/download/{{ file.name }}" class="btn btn-small btn-success">Download</a>
            <a href="/delete/{{ file.name }}" class="btn btn-small btn-danger" onclick="return confirm('Delete this file?');">Delete</a>
        </div>
    </li>
    {% endfor %}
</ul>
{% else %}
<div class="empty-state">
    <div class="icon">📭</div>
    <p style="font-size: 18px;">No files uploaded yet</p>
    <br>
    <a href="/" class="btn">Upload Your First File</a>
</div>
{% endif %}
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📄', 'doc': '📝', 'docx': '📝', 'txt': '📃', 'rtf': '📃',
        'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'ppt': '📽️', 'pptx': '📽️',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️',
        'svg': '🖼️', 'webp': '🖼️', 'ico': '🖼️',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵', 'ogg': '🎵',
        'zip': '🗜️', 'rar': '🗜️', 'tar': '🗜️', 'gz': '🗜️', '7z': '🗜️',
        'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨', 'json': '📋',
    }
    return icons.get(ext, '📎')


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


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
    message = request.args.get('message', '')
    message_class = request.args.get('message_class', '')
    return render_with_base(UPLOAD_TEMPLATE, message=message, message_class=message_class)


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return redirect(url_for('index', message='No files selected', message_class='message-error'))

    files = request.files.getlist('files')
    uploaded_count = 0

    for file in files:
        if file.filename and file.filename.strip():
            filename = file.filename.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_count += 1

    if uploaded_count == 0:
        return redirect(url_for('index', message='No files were uploaded', message_class='message-error'))

    return redirect(url_for('index',
                            message=f'Successfully uploaded {uploaded_count} file(s)!',
                            message_class='message-success'))


@app.route('/files')
def list_files():
    message = request.args.get('message', '')
    message_class = request.args.get('message_class', '')
    upload_dir = app.config['UPLOAD_FOLDER']
    file_list = []

    if os.path.exists(upload_dir):
        for filename in sorted(os.listdir(upload_dir)):
            filepath = os.path.join(upload_dir, filename)
            if os.path.isfile(filepath):
                file_list.append({
                    'name': filename,
                    'size': format_size(os.path.getsize(filepath)),
                    'icon': get_file_icon(filename),
                })

    return render_with_base(FILES_TEMPLATE, files=file_list, message=message, message_class=message_class)


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
        return redirect(url_for('list_files',
                                message=f'Deleted "{filename}"',
                                message_class='message-success'))
    return redirect(url_for('list_files',
                            message=f'File "{filename}" not found',
                            message_class='message-error'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)