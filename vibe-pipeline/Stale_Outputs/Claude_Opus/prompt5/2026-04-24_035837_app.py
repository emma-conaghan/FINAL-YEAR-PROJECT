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
            margin: 0;
            padding: 0;
            box-sizing: border-box;
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
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        nav a {
            color: #ecf0f1;
            text-decoration: none;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 5px;
            transition: background 0.3s;
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
        .card {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            margin-bottom: 25px;
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
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: border-color 0.3s, background 0.3s;
        }
        .upload-area:hover {
            border-color: #3498db;
            background: #f8f9ff;
        }
        .upload-area input[type="file"] {
            margin: 15px 0;
        }
        .btn {
            display: inline-block;
            padding: 10px 24px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 15px;
            text-decoration: none;
            transition: background 0.3s;
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
        .btn-small {
            padding: 6px 14px;
            font-size: 13px;
        }
        .file-list {
            list-style: none;
        }
        .file-list li {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
            transition: background 0.2s;
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
            width: 36px;
            text-align: center;
            flex-shrink: 0;
        }
        .file-name {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .file-size {
            color: #95a5a6;
            font-size: 13px;
            flex-shrink: 0;
            margin-right: 15px;
        }
        .file-actions {
            display: flex;
            gap: 8px;
            flex-shrink: 0;
        }
        .message {
            padding: 12px 20px;
            border-radius: 5px;
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
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <nav>
        <a href="/" class="brand">&#128193; FileVault</a>
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
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <div class="upload-area">
            <div style="font-size: 48px; margin-bottom: 10px;">&#128228;</div>
            <p style="margin-bottom: 15px; color: #7f8c8d;">Select files to upload (max 50MB each)</p>
            <input type="file" name="files" multiple required>
        </div>
        <button type="submit" class="btn btn-success">&#10003; Upload Files</button>
    </form>
</div>

<div class="card">
    <h2>Recent Uploads</h2>
    {% if files %}
    <ul class="file-list">
        {% for file in files[:5] %}
        <li>
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <span class="file-name">{{ file.name }}</span>
            </div>
            <span class="file-size">{{ file.size }}</span>
            <div class="file-actions">
                <a href="/files/{{ file.name }}" class="btn btn-small" target="_blank">Open</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% if files|length > 5 %}
    <div style="text-align: center; margin-top: 15px;">
        <a href="/files" class="btn btn-small">View all {{ files|length }} files</a>
    </div>
    {% endif %}
    {% else %}
    <div class="empty-state">
        <div class="icon">&#128196;</div>
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
    <h1>All Uploaded Files ({{ files|length }})</h1>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li>
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <span class="file-name">{{ file.name }}</span>
            </div>
            <span class="file-size">{{ file.size }}</span>
            <div class="file-actions">
                <a href="/files/{{ file.name }}" class="btn btn-small" target="_blank">Open</a>
                <a href="/download/{{ file.name }}" class="btn btn-small btn-success">Download</a>
                <a href="/delete/{{ file.name }}" class="btn btn-small" style="background:#e74c3c;" onclick="return confirm('Delete this file?')">Delete</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty-state">
        <div class="icon">&#128196;</div>
        <p>No files uploaded yet</p>
        <a href="/" class="btn" style="margin-top: 15px;">Upload Files</a>
    </div>
    {% endif %}
</div>
{% endblock %}
'''


def get_file_icon(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    icon_map = {
        'pdf': '&#128213;',
        'doc': '&#128196;', 'docx': '&#128196;',
        'xls': '&#128202;', 'xlsx': '&#128202;',
        'ppt': '&#128218;', 'pptx': '&#128218;',
        'txt': '&#128196;', 'csv': '&#128196;',
        'png': '&#127912;', 'jpg': '&#127912;', 'jpeg': '&#127912;', 'gif': '&#127912;', 'bmp': '&#127912;', 'svg': '&#127912;', 'webp': '&#127912;',
        'mp4': '&#127916;', 'avi': '&#127916;', 'mov': '&#127916;', 'mkv': '&#127916;',
        'mp3': '&#127925;', 'wav': '&#127925;', 'flac': '&#127925;',
        'zip': '&#128230;', 'rar': '&#128230;', 'tar': '&#128230;', 'gz': '&#128230;',
        'py': '&#128187;', 'js': '&#128187;', 'html': '&#128187;', 'css': '&#128187;',
    }
    return icon_map.get(ext, '&#128196;')


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_files_list():
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


@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', '')
    message_class = request.args.get('message_class', '')
    files = get_files_list()
    return render_template_string(
        UPLOAD_TEMPLATE,
        base=BASE_TEMPLATE,
        files=files,
        message=message,
        message_class=message_class,
    )


@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return redirect(url_for('index', message='No files selected', message_class='message-error'))

    uploaded_files = request.files.getlist('files')
    count = 0
    for f in uploaded_files:
        if f and f.filename and f.filename.strip():
            filename = f.filename.replace('/', '_').replace('\\', '_')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            count += 1

    if count == 0:
        return redirect(url_for('index', message='No valid files were uploaded', message_class='message-error'))

    return redirect(url_for('index', message=f'Successfully uploaded {count} file(s)', message_class='message-success'))


@app.route('/files')
def files_list():
    files = get_files_list()
    return render_template_string(
        FILES_TEMPLATE,
        base=BASE_TEMPLATE,
        files=files,
    )


@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/delete/<path:filename>')
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.isfile(filepath):
        os.remove(filepath)
    return redirect(url_for('files_list'))


# Override Jinja2 to support our pseudo-extends
_original_get_source = None

class PseudoLoader:
    def __init__(self, original_env):
        self.original_env = original_env

    def get_source(self, environment, template):
        if template == "base":
            return BASE_TEMPLATE, "base", lambda: True
        raise Exception(f"Template {template} not found")


@app.before_request
def setup_jinja():
    app.jinja_env.loader = PseudoLoader(app.jinja_env)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)