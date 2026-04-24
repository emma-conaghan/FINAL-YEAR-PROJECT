from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'doc', 'docx', 'xls', 'xlsx', 'csv', 'svg', 'webp', 'mp4', 'mp3', 'zip'}

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
        .navbar nav a { color: #ecf0f1; text-decoration: none; margin-left: 20px; font-weight: 500; transition: color 0.2s; }
        .navbar nav a:hover { color: #3498db; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 10px; padding: 40px; text-align: center; transition: border-color 0.3s, background 0.3s; }
        .upload-area:hover { border-color: #3498db; background: #f7fbff; }
        .upload-area input[type="file"] { margin: 15px 0; }
        .btn { display: inline-block; padding: 10px 24px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 1em; text-decoration: none; transition: background 0.2s; }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .file-list { list-style: none; }
        .file-list li { display: flex; align-items: center; justify-content: space-between; padding: 12px 15px; border-bottom: 1px solid #ecf0f1; transition: background 0.2s; }
        .file-list li:hover { background: #f7f9fc; }
        .file-list li:last-child { border-bottom: none; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 1.5em; }
        .file-name { font-weight: 500; word-break: break-all; }
        .file-size { color: #7f8c8d; font-size: 0.85em; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .file-actions a, .file-actions button { padding: 6px 14px; font-size: 0.85em; border-radius: 4px; }
        .flash-messages { margin-bottom: 15px; }
        .flash { padding: 12px 18px; border-radius: 6px; margin-bottom: 8px; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .empty-state { text-align: center; padding: 40px; color: #95a5a6; }
        .empty-state p { font-size: 1.1em; margin-top: 10px; }
        form.delete-form { display: inline; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📁 File Upload App</h1>
        <nav>
            <a href="/">Upload</a>
            <a href="/files">My Files</a>
        </nav>
    </div>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
        <div class="flash-messages">
            {% for category, message in messages %}
            <div class="flash flash-{{ category }}">{{ message }}</div>
            {% endfor %}
        </div>
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
    <h2>Upload a File</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-area">
            <p style="font-size: 2em; margin-bottom: 10px;">📤</p>
            <p>Select a file to upload</p>
            <input type="file" name="file" id="file" required>
            <br><br>
            <button type="submit" class="btn btn-success">Upload File</button>
        </div>
    </form>
    <p style="margin-top: 15px; color: #7f8c8d; font-size: 0.9em;">
        Allowed file types: {{ allowed_extensions }}. Max size: 16MB.
    </p>
</div>
<div class="card">
    <h2>Recent Uploads</h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files[:5] %}
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
            </div>
        </li>
        {% endfor %}
    </ul>
    {% if files|length > 5 %}
    <p style="text-align: center; margin-top: 15px;"><a href="/files" class="btn">View All Files</a></p>
    {% endif %}
    {% else %}
    <div class="empty-state">
        <p style="font-size: 2em;">📭</p>
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
                <form class="delete-form" method="POST" action="/delete/{{ file.name }}" onsubmit="return confirm('Are you sure you want to delete this file?');">
                    <button type="submit" class="btn btn-danger">Delete</button>
                </form>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <p style="font-size: 2em;">📭</p>
        <p>No files uploaded yet.</p>
        <br>
        <a href="/" class="btn">Upload a File</a>
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
        'pdf': '📄',
        'txt': '📝',
        'doc': '📃', 'docx': '📃',
        'xls': '📊', 'xlsx': '📊', 'csv': '📊',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬',
        'mp3': '🎵',
        'zip': '🗜️',
    }
    return icons.get(ext, '📁')


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


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


from jinja2 import DictLoader

template_loader = DictLoader({
    'base': BASE_TEMPLATE,
    'upload': UPLOAD_TEMPLATE,
    'files': FILES_TEMPLATE,
})

app.jinja_loader = template_loader


@app.route('/')
def index():
    files = get_uploaded_files()
    allowed_ext_str = ', '.join(sorted(ALLOWED_EXTENSIONS))
    return render_template_string(
        "{% extends 'upload' %}",
        files=files,
        allowed_extensions=allowed_ext_str
    )


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected.', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        if not filename:
            flash('Invalid filename.', 'error')
            return redirect(url_for('index'))

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Handle duplicate filenames
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base}_{counter}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            counter += 1

        file.save(filepath)
        flash(f'File "{filename}" uploaded successfully!', 'success')
        return redirect(url_for('index'))
    else:
        flash('File type not allowed.', 'error')
        return redirect(url_for('index'))


@app.route('/files')
def list_files():
    files = get_uploaded_files()
    return render_template_string("{% extends 'files' %}", files=files)


@app.route('/view/<filename>')
def view_file(filename):
    from werkzeug.utils import secure_filename
    safe_filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)


@app.route('/download/<filename>')
def download_file(filename):
    from werkzeug.utils import secure_filename
    safe_filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename, as_attachment=True)


@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    from werkzeug.utils import secure_filename
    safe_filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        flash(f'File "{safe_filename}" deleted successfully.', 'success')
    else:
        flash('File not found.', 'error')
    return redirect(url_for('list_files'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)