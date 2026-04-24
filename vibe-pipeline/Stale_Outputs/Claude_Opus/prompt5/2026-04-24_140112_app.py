from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload Manager</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; }
        .navbar { background: #2c3e50; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; }
        .navbar h1 { color: white; font-size: 1.4em; }
        .navbar nav a { color: #ecf0f1; text-decoration: none; margin-left: 20px; font-size: 0.95em; transition: color 0.2s; }
        .navbar nav a:hover { color: #3498db; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); padding: 30px; margin-bottom: 25px; }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .flash-messages { list-style: none; }
        .flash-messages li { padding: 12px 18px; margin-bottom: 10px; border-radius: 6px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-messages li.error { background: #f8d7da; color: #721c24; border-color: #f5c6cb; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 10px; padding: 40px; text-align: center; margin-bottom: 20px; transition: border-color 0.3s; }
        .upload-area:hover { border-color: #3498db; }
        .upload-area input[type="file"] { margin: 10px 0; }
        .btn { display: inline-block; padding: 10px 24px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.95em; text-decoration: none; transition: background 0.2s; }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .file-list { list-style: none; }
        .file-item { display: flex; align-items: center; justify-content: space-between; padding: 14px 18px; border-bottom: 1px solid #ecf0f1; transition: background 0.2s; }
        .file-item:hover { background: #f8f9fa; }
        .file-item:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 1.5em; flex-shrink: 0; }
        .file-name { font-weight: 500; word-break: break-all; }
        .file-size { color: #7f8c8d; font-size: 0.85em; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .file-actions a, .file-actions button { padding: 6px 14px; font-size: 0.85em; border-radius: 4px; }
        .empty-state { text-align: center; padding: 40px; color: #95a5a6; }
        .empty-state .icon { font-size: 3em; margin-bottom: 10px; }
        form.inline { display: inline; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📁 File Upload Manager</h1>
        <nav>
            <a href="/">Upload</a>
            <a href="/files">My Files</a>
        </nav>
    </div>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
        <ul class="flash-messages">
            {% for category, message in messages %}
            <li class="{{ category }}">{{ message }}</li>
            {% endfor %}
        </ul>
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
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="upload-area">
            <p style="font-size: 2em; margin-bottom: 10px;">📤</p>
            <p style="margin-bottom: 15px; color: #7f8c8d;">Select files to upload (documents, images, etc.)</p>
            <input type="file" name="files" multiple required>
        </div>
        <button type="submit" class="btn btn-success">Upload Files</button>
    </form>
</div>
<div class="card">
    <h2>Recent Files</h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files[:5] %}
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
            </div>
        </li>
        {% endfor %}
    </ul>
    {% if files|length > 5 %}
    <p style="text-align: center; margin-top: 15px;"><a href="/files" class="btn">View All Files ({{ files|length }})</a></p>
    {% endif %}
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet.</p>
    </div>
    {% endif %}
</div>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>All Uploaded Files ({{ files|length }})</h2>
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
                <form class="inline" action="/delete/{{ file.name }}" method="post" onsubmit="return confirm('Delete this file?');">
                    <button type="submit" class="btn btn-danger">Delete</button>
                </form>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet.</p>
        <a href="/" class="btn" style="margin-top: 15px;">Upload Files</a>
    </div>
    {% endif %}
</div>
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icon_map = {
        'pdf': '📄',
        'doc': '📝', 'docx': '📝',
        'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'ppt': '📽️', 'pptx': '📽️',
        'txt': '📃', 'log': '📃', 'md': '📃',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵',
        'zip': '🗜️', 'rar': '🗜️', 'tar': '🗜️', 'gz': '🗜️', '7z': '🗜️',
        'py': '🐍', 'js': '⚡', 'html': '🌐', 'css': '🎨',
        'json': '📋', 'xml': '📋', 'yaml': '📋', 'yml': '📋',
    }
    return icon_map.get(ext, '📎')


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
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                files.append({
                    'name': filename,
                    'size': format_size(size),
                    'icon': get_file_icon(filename),
                })
    return files


# Register base template
@app.before_request
def setup_templates():
    pass


from jinja2 import DictLoader, ChoiceLoader

template_dict = {
    'base': BASE_TEMPLATE,
    'upload.html': UPLOAD_TEMPLATE,
    'files.html': FILES_TEMPLATE,
}

app.jinja_loader = ChoiceLoader([
    DictLoader(template_dict),
    app.jinja_loader,
])


@app.route('/')
def index():
    files = get_uploaded_files()
    return render_template_string(UPLOAD_TEMPLATE, files=files)


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        flash('No files selected.', 'error')
        return redirect(url_for('index'))

    uploaded_files = request.files.getlist('files')
    count = 0
    for file in uploaded_files:
        if file.filename and file.filename.strip():
            filename = file.filename.strip()
            # Simple sanitization - remove path separators
            filename = filename.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            count += 1

    if count > 0:
        flash(f'Successfully uploaded {count} file(s).', 'success')
    else:
        flash('No valid files were uploaded.', 'error')

    return redirect(url_for('index'))


@app.route('/files')
def files_list():
    files = get_uploaded_files()
    return render_template_string(FILES_TEMPLATE, files=files)


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
        flash(f'File "{filename}" has been deleted.', 'success')
    else:
        flash(f'File "{filename}" not found.', 'error')
    return redirect(url_for('files_list'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)