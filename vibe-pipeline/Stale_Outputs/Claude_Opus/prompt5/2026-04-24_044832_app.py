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
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        nav a {
            color: #ecf0f1;
            text-decoration: none;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 6px;
            transition: background 0.2s;
        }
        nav a:hover {
            background: #34495e;
        }
        nav .brand {
            font-weight: bold;
            font-size: 20px;
            color: #3498db;
        }
        .container {
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        h1 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        h2 {
            margin-bottom: 15px;
            color: #2c3e50;
        }
        .upload-area {
            border: 2px dashed #bdc3c7;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            transition: border-color 0.3s, background 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            border-color: #3498db;
            background: #f8f9ff;
        }
        input[type="file"] {
            margin: 15px 0;
        }
        .btn {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            text-decoration: none;
            transition: background 0.2s, transform 0.1s;
        }
        .btn:hover {
            background: #2980b9;
            transform: translateY(-1px);
        }
        .btn:active {
            transform: translateY(0);
        }
        .btn-sm {
            padding: 6px 14px;
            font-size: 13px;
        }
        .btn-green {
            background: #27ae60;
        }
        .btn-green:hover {
            background: #219a52;
        }
        .btn-red {
            background: #e74c3c;
        }
        .btn-red:hover {
            background: #c0392b;
        }
        .file-list {
            list-style: none;
        }
        .file-list li {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 18px;
            border-bottom: 1px solid #ecf0f1;
            transition: background 0.2s;
        }
        .file-list li:last-child {
            border-bottom: none;
        }
        .file-list li:hover {
            background: #f8f9fa;
        }
        .file-name {
            font-size: 15px;
            word-break: break-all;
            flex: 1;
            margin-right: 15px;
        }
        .file-icon {
            margin-right: 10px;
            font-size: 20px;
        }
        .file-actions {
            display: flex;
            gap: 8px;
            flex-shrink: 0;
        }
        .message {
            padding: 12px 20px;
            border-radius: 8px;
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
            color: #95a5a6;
        }
        .empty-state .icon {
            font-size: 48px;
            margin-bottom: 10px;
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
            <p style="font-size: 36px; margin-bottom: 10px;">📤</p>
            <p style="font-size: 18px; margin-bottom: 15px;">Choose files to upload</p>
            <input type="file" name="files" multiple required>
            <p style="color: #95a5a6; font-size: 13px; margin-top: 10px;">Max file size: 50 MB</p>
        </div>
        <button type="submit" class="btn">Upload Files</button>
    </form>
</div>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>Uploaded Files</h1>
    {% if message %}
    <div class="message {{ message_class }}">{{ message }}</div>
    {% endif %}
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li>
            <span class="file-icon">{{ file.icon }}</span>
            <span class="file-name">{{ file.name }}</span>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="btn btn-sm" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-sm btn-green">Download</a>
                <a href="/delete/{{ file.name }}" class="btn btn-sm btn-red" onclick="return confirm('Delete {{ file.name }}?')">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📂</div>
        <p>No files uploaded yet.</p>
        <br>
        <a href="/" class="btn">Upload Your First File</a>
    </div>
    {% endif %}
</div>
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icon_map = {
        'pdf': '📕',
        'doc': '📘', 'docx': '📘',
        'xls': '📗', 'xlsx': '📗',
        'ppt': '📙', 'pptx': '📙',
        'txt': '📄', 'csv': '📄', 'log': '📄',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵',
        'zip': '🗜️', 'rar': '🗜️', 'tar': '🗜️', 'gz': '🗜️',
        'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨',
        'json': '📋', 'xml': '📋',
    }
    return icon_map.get(ext, '📎')


def render(template_str, **kwargs):
    from jinja2 import Environment
    env = Environment()
    base = env.from_string(BASE_TEMPLATE)
    env.globals['base'] = base

    full_template = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', template_str.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''))
    return render_template_string(full_template, **kwargs)


@app.route('/')
def index():
    message = request.args.get('message', '')
    message_class = request.args.get('message_class', '')
    return render(UPLOAD_TEMPLATE, message=message, message_class=message_class)


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return redirect(url_for('index', message='No files selected.', message_class='message-error'))

    files = request.files.getlist('files')
    uploaded_count = 0

    for file in files:
        if file and file.filename and file.filename.strip():
            filename = file.filename.strip()
            filename = filename.replace('..', '').replace('/', '').replace('\\', '')
            if filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_count += 1

    if uploaded_count > 0:
        msg = f'Successfully uploaded {uploaded_count} file(s)!'
        return redirect(url_for('file_list', message=msg, message_class='message-success'))
    else:
        return redirect(url_for('index', message='No valid files to upload.', message_class='message-error'))


@app.route('/files')
def file_list():
    message = request.args.get('message', '')
    message_class = request.args.get('message_class', '')
    filenames = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        filenames = sorted(os.listdir(app.config['UPLOAD_FOLDER']))

    files = []
    for name in filenames:
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
        if os.path.isfile(full_path):
            files.append({
                'name': name,
                'icon': get_file_icon(name),
            })

    return render(FILES_TEMPLATE, files=files, message=message, message_class=message_class)


@app.route('/view/<path:filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/delete/<path:filename>')
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        os.remove(filepath)
        return redirect(url_for('file_list', message=f'Deleted {filename}.', message_class='message-success'))
    return redirect(url_for('file_list', message='File not found.', message_class='message-error'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)