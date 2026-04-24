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
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
        }
        h1 {
            margin-bottom: 25px;
            color: #2c3e50;
        }
        .upload-area {
            background: white;
            border: 2px dashed #3498db;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: border-color 0.2s;
        }
        .upload-area:hover {
            border-color: #2980b9;
        }
        .upload-area input[type="file"] {
            margin: 15px 0;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.2s;
            text-decoration: none;
            display: inline-block;
        }
        .btn:hover {
            background: #2980b9;
        }
        .btn-danger {
            background: #e74c3c;
        }
        .btn-danger:hover {
            background: #c0392b;
        }
        .btn-success {
            background: #27ae60;
        }
        .btn-success:hover {
            background: #219a52;
        }
        .message {
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        .message.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .message.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .file-list {
            list-style: none;
        }
        .file-item {
            background: white;
            border-radius: 8px;
            padding: 18px 24px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            transition: box-shadow 0.2s;
        }
        .file-item:hover {
            box-shadow: 0 3px 12px rgba(0,0,0,0.12);
        }
        .file-name {
            font-size: 16px;
            font-weight: 500;
            word-break: break-all;
            flex: 1;
            margin-right: 15px;
        }
        .file-size {
            color: #7f8c8d;
            font-size: 14px;
            margin-right: 15px;
            white-space: nowrap;
        }
        .file-actions {
            display: flex;
            gap: 8px;
        }
        .file-actions a {
            padding: 8px 16px;
            font-size: 14px;
            border-radius: 5px;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #95a5a6;
        }
        .empty-state p {
            font-size: 18px;
            margin-bottom: 20px;
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
<h1>Upload Files</h1>
{% if message %}
<div class="message {{ message_type }}">{{ message }}</div>
{% endif %}
<div class="upload-area">
    <h2>Drag & drop or select files</h2>
    <p style="color: #7f8c8d; margin-top: 8px;">Upload documents, images, and more (max 50MB)</p>
    <form method="POST" enctype="multipart/form-data" action="/upload">
        <input type="file" name="files" multiple><br>
        <button type="submit" class="btn">Upload Files</button>
    </form>
</div>
<div style="text-align: center;">
    <a href="/files" class="btn btn-success">View All Files →</a>
</div>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Uploaded Files ({{ files|length }})</h1>
{% if files %}
<ul class="file-list">
    {% for file in files %}
    <li class="file-item">
        <span class="file-name">{{ file.name }}</span>
        <span class="file-size">{{ file.size }}</span>
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
    <p>No files uploaded yet.</p>
    <a href="/" class="btn">Upload your first file</a>
</div>
{% endif %}
{% endblock %}
'''


def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def render_with_base(template_str, **kwargs):
    from jinja2 import Environment, BaseLoader, DictLoader
    env = Environment(loader=DictLoader({
        'base': BASE_TEMPLATE,
        'page': template_str
    }))
    template = env.get_template('page')
    return template.render(**kwargs)


@app.route('/')
def index():
    message = request.args.get('message', '')
    message_type = request.args.get('message_type', 'success')
    return render_with_base(UPLOAD_TEMPLATE, message=message, message_type=message_type)


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return redirect(url_for('index', message='No files selected.', message_type='error'))

    files = request.files.getlist('files')
    uploaded_count = 0

    for file in files:
        if file and file.filename and file.filename.strip():
            filename = file.filename
            # Basic security: remove path separators
            filename = filename.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_count += 1

    if uploaded_count == 0:
        return redirect(url_for('index', message='No valid files were uploaded.', message_type='error'))

    return redirect(url_for('index', message=f'Successfully uploaded {uploaded_count} file(s)!', message_type='success'))


@app.route('/files')
def list_files():
    files = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER'])):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                files.append({
                    'name': filename,
                    'size': format_file_size(size)
                })
    return render_with_base(FILES_TEMPLATE, files=files)


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
    return redirect(url_for('list_files'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)