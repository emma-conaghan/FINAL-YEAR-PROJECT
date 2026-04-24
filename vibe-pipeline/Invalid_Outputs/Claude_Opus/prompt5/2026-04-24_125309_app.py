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
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            gap: 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        nav a {
            color: #ecf0f1;
            text-decoration: none;
            font-size: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background 0.2s;
        }
        nav a:hover {
            background: #34495e;
        }
        nav .brand {
            font-size: 1.3rem;
            font-weight: bold;
            color: #3498db;
        }
        .container {
            max-width: 900px;
            margin: 2rem auto;
            padding: 0 1rem;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
        }
        h1, h2 {
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        .upload-area {
            border: 2px dashed #bdc3c7;
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            transition: border-color 0.2s;
        }
        .upload-area:hover {
            border-color: #3498db;
        }
        input[type="file"] {
            margin: 1rem 0;
        }
        .btn {
            display: inline-block;
            background: #3498db;
            color: white;
            border: none;
            padding: 0.7rem 1.5rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1rem;
            text-decoration: none;
            transition: background 0.2s;
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
        .file-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1rem;
            border-bottom: 1px solid #ecf0f1;
            transition: background 0.2s;
        }
        .file-item:hover {
            background: #f8f9fa;
        }
        .file-item:last-child {
            border-bottom: none;
        }
        .file-info {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            flex: 1;
            min-width: 0;
        }
        .file-icon {
            font-size: 1.5rem;
            width: 40px;
            text-align: center;
        }
        .file-name {
            font-weight: 500;
            word-break: break-all;
        }
        .file-size {
            color: #7f8c8d;
            font-size: 0.85rem;
        }
        .file-actions {
            display: flex;
            gap: 0.5rem;
            flex-shrink: 0;
        }
        .file-actions a {
            padding: 0.4rem 0.8rem;
            font-size: 0.85rem;
        }
        .message {
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
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
            padding: 3rem;
            color: #7f8c8d;
        }
        .empty-state .icon {
            font-size: 3rem;
            margin-bottom: 1rem;
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
    <p style="color: #7f8c8d; margin-bottom: 1rem;">Upload documents, images, and other files. Max size: 50MB.</p>
    {% if message %}
    <div class="message {{ message_class }}">{{ message }}</div>
    {% endif %}
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-area">
            <p style="font-size: 2rem; margin-bottom: 0.5rem;">📤</p>
            <p>Select files to upload</p>
            <input type="file" name="files" multiple>
        </div>
        <button type="submit" class="btn btn-success">Upload Files</button>
    </form>
</div>

<div class="card">
    <h2>Recent Files</h2>
    {% if recent_files %}
    <ul class="file-list">
        {% for file in recent_files %}
        <li class="file-item">
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <div>
                    <div class="file-name">{{ file.name }}</div>
                    <div class="file-size">{{ file.size }}</div>
                </div>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="btn" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-success">Download</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📂</div>
        <p>No files uploaded yet</p>
    </div>
    {% endif %}
</div>
{% endblock %}
"""

FILES_TEMPLATE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>All Uploaded Files</h1>
    <p style="color: #7f8c8d; margin-bottom: 1rem;">{{ files|length }} file(s) stored</p>
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
                <a href="/view/{{ file.name }}" class="btn" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-success">Download</a>
                <a href="/delete/{{ file.name }}" class="btn btn-danger" onclick="return confirm('Delete this file?')">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📂</div>
        <p>No files uploaded yet.</p>
        <a href="/" class="btn" style="margin-top: 1rem;">Upload Files</a>
    </div>
    {% endif %}
</div>
{% endblock %}
"""


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


def get_file_list():
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
                    'mtime': os.path.getmtime(filepath),
                })
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return files


from jinja2 import BaseLoader, TemplateNotFound, Environment


class DictLoader(BaseLoader):
    def __init__(self, templates):
        self.templates = templates

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)


templates = {
    'base': BASE_TEMPLATE,
    'upload': UPLOAD_TEMPLATE,
    'files': FILES_TEMPLATE,
}

jinja_env = Environment(loader=DictLoader(templates))


def render(template_name, **kwargs):
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)


@app.route('/')
def index():
    recent_files = get_file_list()[:5]
    return render('upload', recent_files=recent_files, message=None, message_class='')


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        recent_files = get_file_list()[:5]
        return render('upload', recent_files=recent_files, message='No files selected.', message_class='message-error')

    uploaded_files = request.files.getlist('files')
    uploaded_count = 0

    for file in uploaded_files:
        if file.filename and file.filename.strip():
            filename = file.filename.strip()
            filename = filename.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_count += 1

    recent_files = get_file_list()[:5]

    if uploaded_count > 0:
        message = f'Successfully uploaded {uploaded_count} file(s)!'
        message_class = 'message-success'
    else:
        message = 'No valid files were uploaded.'
        message_class = 'message-error'

    return render('upload', recent_files=recent_files, message=message, message_class=message_class)


@app.route('/files')
def file_list():
    files = get_file_list()
    return render('files', files=files)


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
    return redirect(url_for('file_list'))


if __name__ == '__main__':
    app.run(debug=True, host