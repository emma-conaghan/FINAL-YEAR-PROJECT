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
    <title>File Manager</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; min-height: 100vh; }
        .navbar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px 30px; color: white; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .navbar h1 { font-size: 1.5rem; }
        .navbar a { color: white; text-decoration: none; margin-left: 20px; padding: 8px 16px; border-radius: 5px; transition: background 0.3s; }
        .navbar a:hover { background: rgba(255,255,255,0.2); }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; box-shadow: 0 2px 15px rgba(0,0,0,0.08); padding: 30px; margin-bottom: 20px; }
        .card h2 { margin-bottom: 20px; color: #444; }
        .upload-area { border: 2px dashed #667eea; border-radius: 10px; padding: 40px; text-align: center; background: #f8f9ff; margin-bottom: 20px; transition: all 0.3s; }
        .upload-area:hover { border-color: #764ba2; background: #f0f1ff; }
        .upload-area input[type="file"] { margin: 15px 0; }
        .btn { display: inline-block; padding: 10px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1rem; text-decoration: none; transition: transform 0.2s, box-shadow 0.2s; }
        .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
        .btn-danger { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
        .btn-danger:hover { box-shadow: 0 4px 12px rgba(231, 76, 60, 0.4); }
        .btn-sm { padding: 6px 14px; font-size: 0.85rem; }
        .file-list { list-style: none; }
        .file-list li { display: flex; align-items: center; justify-content: space-between; padding: 12px 15px; border-bottom: 1px solid #eee; transition: background 0.2s; }
        .file-list li:hover { background: #f8f9ff; }
        .file-list li:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 1.5rem; }
        .file-name { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .file-size { color: #888; font-size: 0.85rem; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .flash-msg { padding: 12px 20px; border-radius: 5px; margin-bottom: 15px; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .empty-state { text-align: center; padding: 40px; color: #888; }
        .empty-state .icon { font-size: 3rem; margin-bottom: 10px; }
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
'''

UPLOAD_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>Upload Files</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-area">
            <p style="font-size: 2rem; margin-bottom: 10px;">📤</p>
            <p style="margin-bottom: 10px;">Choose files to upload</p>
            <input type="file" name="files" multiple>
            <p style="font-size: 0.85rem; color: #888; margin-top: 10px;">Max file size: 50MB</p>
        </div>
        <button type="submit" class="btn">Upload Files</button>
    </form>
</div>

<div class="card">
    <h2>Recent Files</h2>
    {% if recent_files %}
    <ul class="file-list">
        {% for file in recent_files %}
        <li>
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <span class="file-name">{{ file.name }}</span>
                <span class="file-size">{{ file.size }}</span>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="btn btn-sm" target="_blank">Open</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet</p>
    </div>
    {% endif %}
</div>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>All Files ({{ files|length }})</h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li>
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <span class="file-name">{{ file.name }}</span>
                <span class="file-size">{{ file.size }}</span>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="btn btn-sm" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-sm">Download</a>
                <a href="/delete/{{ file.name }}" class="btn btn-sm btn-danger" onclick="return confirm('Delete this file?')">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet</p>
        <a href="/" class="btn" style="margin-top: 15px;">Upload Files</a>
    </div>
    {% endif %}
</div>
{% endblock %}
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📕', 'doc': '📘', 'docx': '📘', 'txt': '📄',
        'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'webp': '🖼️', 'svg': '🖼️',
        'mp4': '🎬', 'mp3': '🎵',
        'zip': '📦', 'html': '🌐', 'css': '🎨', 'js': '⚡', 'json': '📋', 'xml': '📋',
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

def get_files_info():
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        for filename in sorted(os.listdir(UPLOAD_FOLDER)):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                files.append({
                    'name': filename,
                    'size': format_size(size),
                    'size_bytes': size,
                    'icon': get_file_icon(filename),
                    'mtime': mtime,
                })
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return files

# Custom template loader to handle extends
from jinja2 import BaseLoader, TemplateNotFound

class InMemoryLoader(BaseLoader):
    def __init__(self, templates):
        self.templates = templates

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)

templates_dict = {
    'base': BASE_TEMPLATE,
    'upload': UPLOAD_TEMPLATE,
    'files': FILES_TEMPLATE,
}

app.jinja_loader = InMemoryLoader(templates_dict)

@app.route('/')
def index():
    files = get_files_info()
    recent = files[:5]
    return render_template_string(
        '{% extends "upload" %}',
        recent_files=recent
    )

@app.route('/', methods=['GET'])
def index_get():
    files = get_files_info()
    recent = files[:5]
    return app.jinja_env.get_template('upload').render(
        recent_files=recent,
        get_flashed_messages=lambda with_categories=False: []
    )

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('files')
    uploaded_count = 0

    for file in files:
        if file and file.filename and file.filename.strip():
            filename = file.filename.strip()
            # Basic security: remove path separators
            filename = filename.replace('/', '').replace('\\', '')
            if filename and allowed_file(filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_count += 1
            elif filename:
                flash(f'File type not allowed: {filename}', 'error')

    if uploaded_count > 0:
        flash(f'Successfully uploaded {uploaded_count} file(s)', 'success')
    elif not any(f.filename for f in files):
        flash('No files selected', 'error')

    return redirect(url_for('index'))

@app.route('/files')
def files_list():
    files = get_files_info()
    return app.jinja_env.get_template('files').render(
        files=files,
        get_flashed_messages=lambda with_categories=False: []
    )

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
        flash(f'Deleted: {filename}', 'success')
    else:
        flash(f'File not found: {filename}', 'error')
    return redirect(url_for('files_list'))

# Override the index to properly use Flask's template rendering with flash support
@app.route('/')
def home():
    files = get_files_info()
    recent = files[:5]
    template = app.jinja_env.get_template('upload')
    return render_template_string(UPLOAD_TEMPLATE, recent_files=recent)

# Re-register routes properly
app.view_functions.clear()

@app.route('/', methods=['GET