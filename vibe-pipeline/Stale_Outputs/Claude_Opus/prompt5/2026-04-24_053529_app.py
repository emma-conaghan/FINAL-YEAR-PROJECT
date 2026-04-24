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
    <title>File Upload App</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; }
        .navbar { background: #2c3e50; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; }
        .navbar a { color: white; text-decoration: none; font-size: 16px; margin-left: 20px; }
        .navbar a:hover { text-decoration: underline; }
        .navbar .brand { font-size: 22px; font-weight: bold; color: #ecf0f1; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 25px; }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 10px; padding: 40px; text-align: center; transition: border-color 0.3s; }
        .upload-area:hover { border-color: #3498db; }
        .upload-area input[type="file"] { margin: 15px 0; }
        .btn { display: inline-block; padding: 10px 25px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; text-decoration: none; }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .file-list { list-style: none; }
        .file-list li { display: flex; align-items: center; justify-content: space-between; padding: 12px 15px; border-bottom: 1px solid #ecf0f1; }
        .file-list li:last-child { border-bottom: none; }
        .file-list li:hover { background: #f8f9fa; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 28px; flex-shrink: 0; }
        .file-name { font-weight: 500; word-break: break-all; }
        .file-size { color: #7f8c8d; font-size: 13px; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .file-actions a { padding: 6px 14px; font-size: 13px; }
        .flash-msg { padding: 12px 20px; border-radius: 5px; margin-bottom: 20px; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .empty-state { text-align: center; padding: 50px; color: #95a5a6; }
        .empty-state .icon { font-size: 60px; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="navbar">
        <span class="brand">📁 File Upload App</span>
        <div>
            <a href="/">Home</a>
            <a href="/upload">Upload</a>
            <a href="/files">Files</a>
        </div>
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

HOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card" style="text-align: center; padding: 50px;">
    <h1 style="margin-bottom: 15px;">Welcome to File Upload App</h1>
    <p style="color: #7f8c8d; margin-bottom: 30px; font-size: 17px;">Upload documents and images, then view or download them anytime.</p>
    <div style="display: flex; gap: 15px; justify-content: center;">
        <a href="/upload" class="btn btn-success" style="font-size: 17px; padding: 12px 30px;">⬆ Upload Files</a>
        <a href="/files" class="btn" style="font-size: 17px; padding: 12px 30px;">📄 View Files</a>
    </div>
</div>
{% endblock %}
'''

UPLOAD_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>⬆ Upload Files</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-area">
            <p style="font-size: 18px; margin-bottom: 10px;">Choose files to upload</p>
            <p style="color: #95a5a6; margin-bottom: 15px;">Supports documents, images, and other file types (max 50MB)</p>
            <input type="file" name="files" multiple required>
            <br><br>
            <button type="submit" class="btn btn-success">Upload Files</button>
        </div>
    </form>
</div>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>📄 Uploaded Files ({{ files|length }})</h2>
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
                <a href="/view/{{ file.name }}" class="btn" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-success">Download</a>
                <a href="/delete/{{ file.name }}" class="btn btn-danger" onclick="return confirm('Delete {{ file.name }}?');">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <h3>No files uploaded yet</h3>
        <p style="margin-top: 10px;">
            <a href="/upload" class="btn btn-success">Upload your first file</a>
        </p>
    </div>
    {% endif %}
</div>
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📕', 'doc': '📘', 'docx': '📘', 'txt': '📝', 'rtf': '📝',
        'xls': '📗', 'xlsx': '📗', 'csv': '📗',
        'ppt': '📙', 'pptx': '📙',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'zip': '📦', 'rar': '📦', 'tar': '📦', 'gz': '📦', '7z': '📦',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵',
        'mp4': '🎬', 'avi': '🎬', 'mkv': '🎬', 'mov': '🎬',
        'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨', 'json': '📋',
    }
    return icons.get(ext, '📄')


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


from jinja2 import DictLoader

template_loader = DictLoader({
    'base': BASE_TEMPLATE,
    'home': HOME_TEMPLATE,
    'upload': UPLOAD_TEMPLATE,
    'files': FILES_TEMPLATE,
})


app.jinja_loader = template_loader


@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No files selected.', 'error')
            return redirect(url_for('upload'))

        files = request.files.getlist('files')
        uploaded_count = 0

        for f in files:
            if f and f.filename and f.filename.strip():
                filename = f.filename.strip()
                # Basic sanitization - remove path separators
                filename = filename.replace('/', '_').replace('\\', '_')
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(filepath)
                uploaded_count += 1

        if uploaded_count > 0:
            flash(f'Successfully uploaded {uploaded_count} file(s)!', 'success')
            return redirect(url_for('files'))
        else:
            flash('No valid files were uploaded.', 'error')
            return redirect(url_for('upload'))

    return render_template_string(UPLOAD_TEMPLATE)


@app.route('/files')
def files():
    file_list = get_uploaded_files()
    return render_template_string(FILES_TEMPLATE, files=file_list)


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
        flash(f'File "{filename}" has been deleted.', 'success')
    else:
        flash(f'File "{filename}" not found.', 'error')
    return redirect(url_for('files'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)