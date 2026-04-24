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
            padding: 15px 30px;
            display: flex;
            align-items: center;
            gap: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
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
        .upload-form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .file-input-wrapper {
            border: 2px dashed #bdc3c7;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.3s, background 0.3s;
            position: relative;
        }
        .file-input-wrapper:hover {
            border-color: #3498db;
            background: #f7fbff;
        }
        .file-input-wrapper input[type="file"] {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: pointer;
        }
        .file-input-wrapper p {
            font-size: 16px;
            color: #7f8c8d;
        }
        .file-input-wrapper .icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.2s;
            text-decoration: none;
            display: inline-block;
            text-align: center;
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
            font-size: 13px;
            padding: 6px 12px;
        }
        .btn-danger:hover {
            background: #c0392b;
        }
        .btn-small {
            font-size: 13px;
            padding: 6px 12px;
        }
        .file-list {
            list-style: none;
        }
        .file-list li {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            border-bottom: 1px solid #ecf0f1;
            transition: background 0.15s;
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
            min-width: 0;
        }
        .file-icon {
            font-size: 24px;
            flex-shrink: 0;
        }
        .file-name {
            font-size: 15px;
            color: #2c3e50;
            word-break: break-all;
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
            font-size: 15px;
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
            font-size: 64px;
            margin-bottom: 15px;
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

UPLOAD_PAGE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>Upload Files</h1>
    {% if message %}
        <div class="message {{ message_class }}">{{ message }}</div>
    {% endif %}
    <form class="upload-form" method="POST" action="/upload" enctype="multipart/form-data">
        <div class="file-input-wrapper" id="dropZone">
            <div class="icon">📤</div>
            <p id="fileLabel">Click or drag files here to upload</p>
            <input type="file" name="file" id="fileInput" onchange="updateLabel(this)">
        </div>
        <button type="submit" class="btn btn-success">Upload File</button>
    </form>
</div>
<script>
function updateLabel(input) {
    var label = document.getElementById('fileLabel');
    if (input.files && input.files.length > 0) {
        label.textContent = 'Selected: ' + input.files[0].name;
    } else {
        label.textContent = 'Click or drag files here to upload';
    }
}
var dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    dropZone.style.borderColor = '#3498db';
    dropZone.style.background = '#f7fbff';
});
dropZone.addEventListener('dragleave', function(e) {
    e.preventDefault();
    dropZone.style.borderColor = '#bdc3c7';
    dropZone.style.background = '';
});
dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    dropZone.style.borderColor = '#bdc3c7';
    dropZone.style.background = '';
    var input = document.getElementById('fileInput');
    input.files = e.dataTransfer.files;
    updateLabel(input);
});
</script>
{% endblock %}
"""

FILES_PAGE = """
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>Uploaded Files</h1>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li>
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <span class="file-name">{{ file.name }}</span>
            </div>
            <div class="file-actions">
                <a href="/view/{{ file.name }}" class="btn btn-small" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-small btn-success">Download</a>
                <a href="/delete/{{ file.name }}" class="btn btn-small btn-danger" onclick="return confirm('Delete this file?')">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">📭</div>
        <p>No files uploaded yet.</p>
        <br>
        <a href="/" class="btn">Upload your first file</a>
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
        'zip': '🗜️', 'rar': '🗜️', 'tar': '🗜️', 'gz': '🗜️',
        'py': '🐍', 'js': '📜', 'html': '🌐', 'css': '🎨',
        'json': '📋', 'xml': '📋',
    }
    return icons.get(ext, '📁')


def render_with_base(template_string, **kwargs):
    from jinja2 import Environment, BaseLoader
    env = Environment(loader=BaseLoader())
    base_template = env.from_string(BASE_TEMPLATE)
    env.globals['base'] = base_template

    class CustomTemplate:
        pass

    full_template_str = template_string.replace('{% extends "base" %}', '')
    block_content = ''
    import re
    match = re.search(r'{%\s*block content\s*%}(.*?){%\s*endblock\s*%}', full_template_str, re.DOTALL)
    if match:
        block_content = match.group(1)

    final_str = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', block_content)
    template = env.from_string(final_str)
    return template.render(**kwargs)


@app.route('/')
def index():
    return render_with_base(UPLOAD_PAGE, message=request.args.get('message', ''),
                            message_class=request.args.get('message_class', ''))


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file selected.', message_class='message-error'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', message='No file selected.', message_class='message-error'))

    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    if not filename:
        return redirect(url_for('index', message='Invalid filename.', message_class='message-error'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Handle duplicate filenames
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filepath):
        filename = f"{base}_{counter}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        counter += 1

    file.save(filepath)
    return redirect(url_for('index', message=f'File "{filename}" uploaded successfully!',
                            message_class='message-success'))


@app.route('/files')
def files():
    file_list = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for name in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            if os.path.isfile(full_path):
                file_list.append({
                    'name': name,
                    'icon': get_file_icon(name),
                })
    return render_with_base(FILES_PAGE, files=file_list)


@app.route('/view/<filename>')
def view_file(filename):
    from werkzeug.utils import secure_filename
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download/<filename>')
def download_file(filename):
    from werkzeug.utils import secure_filename
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/delete/<filename>')
def delete_file(filename):
    from werkzeug.utils import secure_filename
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return redirect(url_for('files'))


if __name__ == '__main__':
    app.run(debug=True, port