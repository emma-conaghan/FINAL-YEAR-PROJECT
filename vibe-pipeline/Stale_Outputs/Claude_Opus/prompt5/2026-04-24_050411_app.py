from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

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
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .navbar h1 { font-size: 1.4em; }
        .navbar a { color: #ecf0f1; text-decoration: none; margin-left: 20px; font-size: 0.95em; transition: color 0.2s; }
        .navbar a:hover { color: #3498db; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; padding: 30px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 8px; padding: 40px; text-align: center; transition: border-color 0.3s; }
        .upload-area:hover { border-color: #3498db; }
        .upload-area input[type="file"] { margin: 15px 0; }
        .btn { display: inline-block; padding: 10px 25px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.95em; text-decoration: none; transition: background 0.2s; }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .file-list { list-style: none; }
        .file-item { display: flex; align-items: center; justify-content: space-between; padding: 12px 15px; border-bottom: 1px solid #ecf0f1; transition: background 0.2s; }
        .file-item:hover { background: #f8f9fa; }
        .file-item:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 1.5em; flex-shrink: 0; }
        .file-name { font-weight: 500; word-break: break-all; }
        .file-size { color: #7f8c8d; font-size: 0.85em; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; margin-left: 10px; }
        .file-actions a { padding: 6px 14px; font-size: 0.85em; border-radius: 4px; text-decoration: none; color: white; }
        .file-actions .view-btn { background: #3498db; }
        .file-actions .view-btn:hover { background: #2980b9; }
        .file-actions .download-btn { background: #27ae60; }
        .file-actions .download-btn:hover { background: #219a52; }
        .file-actions .delete-btn { background: #e74c3c; }
        .file-actions .delete-btn:hover { background: #c0392b; }
        .flash-msg { padding: 12px 20px; border-radius: 6px; margin-bottom: 15px; font-size: 0.95em; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .empty-state { text-align: center; padding: 40px; color: #95a5a6; }
        .empty-state p { font-size: 1.1em; margin-top: 10px; }
        @media (max-width: 600px) {
            .file-item { flex-direction: column; align-items: flex-start; gap: 10px; }
            .file-actions { margin-left: 0; }
        }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📁 File Upload App</h1>
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

INDEX_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card" style="text-align: center;">
    <h2>Welcome to File Upload App</h2>
    <p style="margin-bottom: 25px; color: #7f8c8d; line-height: 1.6;">
        Upload documents and images to the server, then view or download them anytime.
    </p>
    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
        <a href="/upload" class="btn">📤 Upload Files</a>
        <a href="/files" class="btn btn-success">📋 View Files</a>
    </div>
</div>
<div class="card">
    <h2>Quick Stats</h2>
    <p style="color: #7f8c8d;">Total files uploaded: <strong>{{ file_count }}</strong></p>
</div>
{% endblock %}
'''

UPLOAD_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>📤 Upload Files</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="upload-area">
            <p style="font-size: 2em; margin-bottom: 10px;">📂</p>
            <p style="margin-bottom: 15px; color: #7f8c8d;">Select files to upload (max 16MB each)</p>
            <input type="file" name="files" multiple required>
            <br><br>
            <button type="submit" class="btn">Upload</button>
        </div>
    </form>
</div>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>📋 Uploaded Files</h2>
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
                <a href="/view/{{ file.name }}" class="view-btn" target="_blank">View</a>
                <a href="/download/{{ file.name }}" class="download-btn">Download</a>
                <a href="/delete/{{ file.name }}" class="delete-btn" onclick="return confirm('Delete {{ file.name }}?');">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <p style="font-size: 3em;">📭</p>
        <p>No files uploaded yet.</p>
        <br>
        <a href="/upload" class="btn">Upload your first file</a>
    </div>
    {% endif %}
</div>
{% endblock %}
'''

def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icons = {
        'pdf': '📄', 'doc': '📝', 'docx': '📝', 'txt': '📝',
        'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'ppt': '📽️', 'pptx': '📽️',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️',
        'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬',
        'mp3': '🎵', 'wav': '🎵',
        'zip': '📦', 'rar': '📦', 'tar': '📦', 'gz': '📦',
        'py': '🐍', 'js': '⚡', 'html': '🌐', 'css': '🎨',
    }
    return icons.get(ext, '📎')

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def get_files_info():
    files = []
    upload_folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_folder):
        for filename in sorted(os.listdir(upload_folder)):
            filepath = os.path.join(upload_folder, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                files.append({
                    'name': filename,
                    'size': format_size(size),
                    'icon': get_file_icon(filename)
                })
    return files

@app.route('/')
def index():
    file_count = len(get_files_info())
    return render_template_string(INDEX_TEMPLATE, file_count=file_count, base=BASE_TEMPLATE)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No files selected.', 'error')
            return redirect(url_for('upload'))
        
        files = request.files.getlist('files')
        uploaded_count = 0
        
        for file in files:
            if file and file.filename and file.filename.strip():
                filename = file.filename.replace('/', '_').replace('\\', '_')
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_count += 1
        
        if uploaded_count > 0:
            flash(f'Successfully uploaded {uploaded_count} file(s).', 'success')
        else:
            flash('No valid files were uploaded.', 'error')
        
        return redirect(url_for('files_list'))
    
    return render_template_string(UPLOAD_TEMPLATE, base=BASE_TEMPLATE)

@app.route('/files')
def files_list():
    files = get_files_info()
    return render_template_string(FILES_TEMPLATE, files=files, base=BASE_TEMPLATE)

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
        flash(f'Deleted "{filename}".', 'success')
    else:
        flash(f'File "{filename}" not found.', 'error')
    return redirect(url_for('files_list'))

# Override Jinja2 to support our pseudo-extends
_original_render = render_template_string

def patched_render_template_string(source, **context):
    if '{% extends "base" %}' in source:
        source = source.replace('{% extends "base" %}', '')
        full = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', source.replace('{% block content %}', '').replace('{% endblock %}', ''))
        return _original_render(full, **context)
    return _original_render(source, **context)

import flask
flask.render_template_string = patched_render_template_string
render_template_string = patched_render_template_string

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)