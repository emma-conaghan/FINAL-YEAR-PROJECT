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
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
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
            margin-right: auto;
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
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 28px;
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 22px;
        }
        .upload-area {
            border: 3px dashed #bdc3c7;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: border-color 0.3s, background 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            border-color: #3498db;
            background: #f8f9ff;
        }
        .upload-area input[type="file"] {
            margin: 15px 0;
            font-size: 14px;
        }
        .btn {
            display: inline-block;
            padding: 12px 30px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            text-decoration: none;
            transition: background 0.3s, transform 0.1s;
        }
        .btn:hover {
            background: #2980b9;
            transform: translateY(-1px);
        }
        .btn:active {
            transform: translateY(0);
        }
        .btn-success {
            background: #27ae60;
        }
        .btn-success:hover {
            background: #219a52;
        }
        .btn-small {
            padding: 6px 14px;
            font-size: 13px;
        }
        .file-list {
            list-style: none;
        }
        .file-list li {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 15px 20px;
            border-bottom: 1px solid #ecf0f1;
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
            font-size: 15px;
            color: #2c3e50;
            word-break: break-all;
        }
        .file-size {
            font-size: 13px;
            color: #95a5a6;
        }
        .file-actions {
            display: flex;
            gap: 8px;
        }
        .message {
            padding: 15px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 15px;
        }
        .message-success {
            background: #d5f5e3;
            color: #1e8449;
            border: 1px solid #a9dfbf;
        }
        .message-error {
            background: #fadbd8;
            color: #c0392b;
            border: 1px solid #f1948a;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #95a5a6;
        }
        .empty-state .icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .delete-form {
            display: inline;
        }
        .btn-danger {
            background: #e74c3c;
        }
        .btn-danger:hover {
            background: #c0392b;
        }
    </style>
</head>
<body>
    <nav>
        <span class="brand">📁 FileVault</span>
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
<div class="card">
    <h1>Upload Files</h1>
    {% if message %}
    <div class="message {{ message_class }}">{{ message }}</div>
    {% endif %}
    <form method="POST" enctype="multipart/form-data" action="/upload">
        <div class="upload-area">
            <div style="font-size: 48px; margin-bottom: 10px;">📤</div>
            <p style="font-size: 18px; color: #7f8c8d; margin-bottom: 15px;">Choose a file to upload</p>
            <input type="file" name="file" required>
        </div>
        <div style="text-align: center;">
            <button type="submit" class="btn btn-success">Upload File</button>
        </div>
    </form>
</div>
<div class="card">
    <p style="color: #7f8c8d; font-size: 14px;">
        Supported: Documents, images, PDFs, and more. Max file size: 50 MB.
    </p>
</div>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>My Files</h1>
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
                <a href="/view/{{ file.name }}" class="btn btn-small" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-small btn-success">Download</a>
                <form class="delete-form" method="POST" action="/delete/{{ file.name }}" onsubmit="return confirm('Delete this file?');">
                    <button type="submit" class="btn btn-small btn-danger">Delete</button>
                </form>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p style="font-size: 18px; margin-bottom: 10px;">No files uploaded yet</p>
        <p>Go to the <a href="/" style="color: #3498db;">upload page</a> to add files.</p>
    </div>
    {% endif %}
</div>
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📕',
        'doc': '📘', 'docx': '📘',
        'xls': '📗', 'xlsx': '📗',
        'ppt': '📙', 'pptx': '📙',
        'txt': '📄', 'csv': '📄', 'log': '📄',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵',
        'zip': '🗜️', 'rar': '🗜️', 'tar': '🗜️', 'gz': '🗜️', '7z': '🗜️',
        'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨',
        'json': '📋', 'xml': '📋',
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
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file selected.', message_class='message-error'))

    file = request.files['file']

    if file.filename == '' or file.filename is None:
        return redirect(url_for('index', message='No file selected.', message_class='message-error'))

    filename = file.filename
    # Basic sanitization
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    if not filename:
        return redirect(url_for('index', message='Invalid filename.', message_class='message-error'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    return redirect(url_for('index', message=f'File "{filename}" uploaded successfully!', message_class='message-success'))


@app.route('/files')
def files():
    message = request.args.get('message', '')
    message_class = request.args.get('message_class', '')

    file_list = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for fname in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            if os.path.isfile(fpath):
                file_list.append({
                    'name': fname,
                    'size': format_size(os.path.getsize(fpath)),
                    'icon': get_file_icon(fname),
                })

    return render_with_base(FILES_TEMPLATE, files=file_list, message=message, message_class=message_class)


@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        os.remove(filepath)
        return redirect(url_for('files', message=f'File "{filename}" deleted.', message_class='message-success'))
    return redirect(url_for('files', message='File not found.', message_class='message-error'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)