from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'zip', 'mp4', 'webm', 'mp3', 'html', 'css', 'js', 'json', 'xml'}

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; min-height: 100vh; }
        .navbar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .navbar a { color: white; text-decoration: none; margin-right: 2rem; font-weight: 500; font-size: 1.1rem; transition: opacity 0.2s; }
        .navbar a:hover { opacity: 0.8; }
        .navbar .brand { font-size: 1.4rem; font-weight: 700; }
        .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 2px 15px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
        .card h2 { margin-bottom: 1.5rem; color: #444; }
        .upload-area { border: 2px dashed #667eea; border-radius: 12px; padding: 3rem; text-align: center; background: #f8f9ff; transition: all 0.3s; cursor: pointer; }
        .upload-area:hover { border-color: #764ba2; background: #f0f1ff; }
        .upload-area p { color: #666; margin-bottom: 1rem; font-size: 1.1rem; }
        input[type="file"] { margin: 1rem 0; }
        .btn { display: inline-block; padding: 0.7rem 1.5rem; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; font-weight: 600; text-decoration: none; transition: all 0.2s; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102,126,234,0.4); }
        .btn-download { background: #28a745; color: white; padding: 0.4rem 0.8rem; font-size: 0.85rem; border-radius: 6px; }
        .btn-download:hover { background: #218838; }
        .btn-open { background: #17a2b8; color: white; padding: 0.4rem 0.8rem; font-size: 0.85rem; border-radius: 6px; }
        .btn-open:hover { background: #138496; }
        .btn-delete { background: #dc3545; color: white; padding: 0.4rem 0.8rem; font-size: 0.85rem; border-radius: 6px; border: none; cursor: pointer; }
        .btn-delete:hover { background: #c82333; }
        .file-list { list-style: none; }
        .file-item { display: flex; align-items: center; justify-content: space-between; padding: 1rem; border-bottom: 1px solid #eee; transition: background 0.2s; }
        .file-item:hover { background: #f8f9fa; }
        .file-item:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 0.75rem; flex: 1; min-width: 0; }
        .file-icon { font-size: 1.5rem; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; background: #f0f2f5; border-radius: 8px; }
        .file-name { font-weight: 500; word-break: break-all; }
        .file-size { color: #888; font-size: 0.85rem; }
        .file-actions { display: flex; gap: 0.5rem; flex-shrink: 0; margin-left: 1rem; }
        .flash-msg { padding: 1rem; border-radius: 8px; margin-bottom: 1rem; font-weight: 500; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .empty-state { text-align: center; padding: 3rem; color: #888; }
        .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="/" class="brand">📁 File Manager</a>
        <a href="/">Upload</a>
        <a href="/files">Files</a>
    </nav>
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
    <h2>Upload Files</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="upload-area" onclick="document.getElementById('fileInput').click();">
            <p>📤 Click to select files or use the button below</p>
            <input type="file" name="files" id="fileInput" multiple style="display:none;" onchange="updateLabel(this)">
            <input type="file" name="files_visible" id="fileInputVisible" multiple onchange="syncFiles(this)">
            <p id="fileLabel" style="color: #667eea; font-weight: 600; margin-top: 0.5rem;"></p>
        </div>
        <br>
        <button type="submit" class="btn btn-primary">Upload Files</button>
    </form>
</div>
<script>
function updateLabel(input) {
    var label = document.getElementById('fileLabel');
    if (input.files.length > 0) {
        var names = [];
        for (var i = 0; i < input.files.length; i++) names.push(input.files[i].name);
        label.textContent = names.join(', ');
    }
}
function syncFiles(input) {
    document.getElementById('fileInput').files = input.files;
    updateLabel(input);
}
</script>
{% endblock %}
"""

FILES_PAGE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>Uploaded Files ({{ files|length }})</h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li class="file-item">
            <div class="file-info">
                <div class="file-icon">{{ file.icon }}</div>
                <div>
                    <div class="file-name">{{ file.name }}</div>
                    <div class="file-size">{{ file.size }}</div>
                </div>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="btn btn-open" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-download">Download</a>
                <form action="/delete/{{ file.name }}" method="post" style="display:inline;" onsubmit="return confirm('Delete this file?');">
                    <button type="submit" class="btn-delete">Delete</button>
                </form>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet.</p>
        <br>
        <a href="/" class="btn btn-primary">Upload Files</a>
    </div>
    {% endif %}
</div>
{% endblock %}
"""


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📄', 'doc': '📝', 'docx': '📝', 'txt': '📃',
        'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️',
        'mp4': '🎬', 'webm': '🎬', 'mp3': '🎵',
        'zip': '🗜️', 'html': '🌐', 'css': '🎨', 'js': '⚡', 'json': '📋', 'xml': '📋',
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
    from jinja2 import Environment, BaseLoader
    env = Environment(loader=BaseLoader())
    env.globals.update(get_flashed_messages=lambda with_categories=False: [])
    base = app.jinja_env.from_string(BASE_TEMPLATE)
    app.jinja_env.globals['base'] = base
    combined = BASE_TEMPLATE.replace('{% block content %}{% endblock %}',
                                      template_string.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''))
    return render_template_string(combined, **kwargs)


@app.route('/')
def index():
    combined = BASE_TEMPLATE.replace(
        '{% block content %}{% endblock %}',
        """
<div class="card">
    <h2>Upload Files</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="upload-area">
            <p>📤 Select files to upload</p>
            <input type="file" name="files" id="fileInput" multiple>
        </div>
        <br>
        <button type="submit" class="btn btn-primary">Upload Files</button>
    </form>
</div>
        """
    )
    return render_template_string(combined, title="Upload Files")


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        flash('No files selected.', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('files')
    uploaded_count = 0

    for f in files:
        if f and f.filename and f.filename.strip():
            filename = f.filename.replace('..', '').replace('/', '').replace('\\', '')
            if filename:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(filepath)
                uploaded_count += 1

    if uploaded_count > 0:
        flash(f'Successfully uploaded {uploaded_count} file(s).', 'success')
    else:
        flash('No valid files were uploaded.', 'error')

    return redirect(url_for('file_list'))


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

    combined = BASE_TEMPLATE.replace(
        '{% block content %}{% endblock %}',
        """
<div class="card">
    <h2>Uploaded Files ({{ files|length }})</h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li class="file-item">
            <div class="file-info">
                <div class="file-icon">{{ file.icon }}</div>
                <div>
                    <div class="file-name">{{ file.name }}</div>
                    <div class="file-size">{{ file.size }}</div>
                </div>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="btn btn-open" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-download">Download</a>
                <form action="/delete/{{ file.name }}" method="post" style="display:inline;" onsubmit="return confirm('Delete this file?');">
                    <button type="submit" class="btn-delete">Delete</button>
                </form>
            </div>
        </li>
        {% endfor