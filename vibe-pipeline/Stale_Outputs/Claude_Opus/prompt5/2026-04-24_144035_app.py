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
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; min-height: 100vh; }
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .navbar h1 { font-size: 1.4em; }
        .navbar a { color: white; text-decoration: none; margin-left: 20px; padding: 8px 16px; border-radius: 4px; transition: background 0.2s; }
        .navbar a:hover { background: rgba(255,255,255,0.1); }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }
        .card h2 { margin-bottom: 20px; color: #2c3e50; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 8px; padding: 40px; text-align: center; transition: border-color 0.3s, background 0.3s; cursor: pointer; position: relative; }
        .upload-area:hover { border-color: #3498db; background: #ecf5ff; }
        .upload-area input[type="file"] { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
        .upload-area p { font-size: 1.1em; color: #7f8c8d; margin-bottom: 10px; }
        .upload-area .icon { font-size: 3em; margin-bottom: 10px; }
        .btn { display: inline-block; padding: 10px 24px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; text-decoration: none; transition: background 0.2s; }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .file-list { list-style: none; }
        .file-item { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid #ecf0f1; transition: background 0.2s; }
        .file-item:last-child { border-bottom: none; }
        .file-item:hover { background: #f8f9fa; }
        .file-info { display: flex; align-items: center; gap: 12px; flex: 1; min-width: 0; }
        .file-icon { font-size: 1.5em; flex-shrink: 0; }
        .file-name { font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .file-size { color: #95a5a6; font-size: 0.85em; }
        .file-actions { display: flex; gap: 8px; flex-shrink: 0; }
        .file-actions a { padding: 6px 14px; font-size: 0.85em; border-radius: 4px; text-decoration: none; color: white; }
        .file-actions .view-btn { background: #3498db; }
        .file-actions .view-btn:hover { background: #2980b9; }
        .file-actions .download-btn { background: #27ae60; }
        .file-actions .download-btn:hover { background: #219a52; }
        .file-actions .delete-btn { background: #e74c3c; }
        .file-actions .delete-btn:hover { background: #c0392b; }
        .flash-msg { padding: 12px 20px; border-radius: 4px; margin-bottom: 15px; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .empty-state { text-align: center; padding: 40px; color: #95a5a6; }
        .empty-state .icon { font-size: 3em; margin-bottom: 10px; }
        .file-count { background: #ecf0f1; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; color: #7f8c8d; }
        #file-name-display { margin-top: 10px; font-weight: 500; color: #2c3e50; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📁 File Upload App</h1>
        <div>
            <a href="/">Upload</a>
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

UPLOAD_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h2>Upload a File</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data" id="upload-form">
        <div class="upload-area" id="upload-area">
            <input type="file" name="file" id="file-input" required>
            <div class="icon">📤</div>
            <p>Click or drag a file here to upload</p>
            <p style="font-size: 0.85em; color: #bdc3c7;">Max file size: 16MB</p>
            <div id="file-name-display"></div>
        </div>
        <br>
        <button type="submit" class="btn btn-success">Upload File</button>
    </form>
</div>
<script>
    const fileInput = document.getElementById('file-input');
    const display = document.getElementById('file-name-display');
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            display.textContent = 'Selected: ' + this.files[0].name;
        } else {
            display.textContent = '';
        }
    });
</script>
{% endblock %}
'''

FILES_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <h2 style="margin-bottom: 0;">Uploaded Files <span class="file-count">{{ files|length }}</span></h2>
        <a href="/" class="btn">Upload New</a>
    </div>
    {% if files %}
    <ul class="file-list">
        {% for file in files %}
        <li class="file-item">
            <div class="file-info">
                <span class="file-icon">{{ file.icon }}</span>
                <div>
                    <div class="file-name" title="{{ file.name }}">{{ file.name }}</div>
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
        <div class="icon">📭</div>
        <p>No files uploaded yet.</p>
        <br>
        <a href="/" class="btn">Upload your first file</a>
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
        'pdf': '📕',
        'txt': '📝',
        'doc': '📘', 'docx': '📘',
        'xls': '📗', 'xlsx': '📗', 'csv': '📗',
        'png': '🖼️', 'jpg': '🖼️', 'jpeg': '🖼️', 'gif': '🖼️', 'bmp': '🖼️', 'svg': '🖼️', 'webp': '🖼️',
        'mp4': '🎬',
        'mp3': '🎵',
        'zip': '📦',
    }
    return icons.get(ext, '📄')

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def get_uploaded_files():
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        for filename in sorted(os.listdir(UPLOAD_FOLDER)):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
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
    app.jinja_env.loader = JinjaLoader()

class JinjaLoader:
    def __init__(self):
        self.templates = {
            'base': BASE_TEMPLATE,
            'upload': UPLOAD_TEMPLATE,
            'files': FILES_TEMPLATE,
        }

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise Exception(f"Template {template} not found")

    def list_templates(self):
        return list(self.templates.keys())

from jinja2 import BaseLoader as Jinja2BaseLoader

class JinjaLoader(Jinja2BaseLoader):
    def __init__(self):
        self.templates = {
            'base': BASE_TEMPLATE,
            'upload': UPLOAD_TEMPLATE,
            'files': FILES_TEMPLATE,
        }

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        from jinja2 import TemplateNotFound
        raise TemplateNotFound(template)

@app.route('/')
def index():
    return render_template_string(UPLOAD_TEMPLATE.replace('{% extends "base" %}', BASE_TEMPLATE.replace('{% block content %}{% endblock %}', UPLOAD_TEMPLATE.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''))))

@app.route('/files')
def files():
    uploaded_files = get_uploaded_files()
    template = BASE_TEMPLATE.replace(
        '{% block content %}{% endblock %}',
        FILES_TEMPLATE.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', '')
    )
    return render_template_string(template, files=uploaded_files)

@app.route('/upload', methods=['POST'])
def upload():
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
        return redirect(url_for('files'))
    else:
        flash('File type not allowed.', 'error')
        return redirect(url_for('index'))

@app.route('/view/<filename>')
def view_file(filename):
    from werkzeug.utils import secure_filename
    filename = secure_filename(filename)