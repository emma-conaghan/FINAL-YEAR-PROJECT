import os
import uuid
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, abort

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp',
    'pdf', 'txt', 'doc', 'docx', 'xls', 'xlsx',
    'ppt', 'pptx', 'csv', 'zip', 'mp4', 'mp3'
}

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Uploader</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; color: #333; }
        header { background: #4a90e2; color: white; padding: 16px 32px; display: flex; align-items: center; justify-content: space-between; }
        header h1 { font-size: 1.5rem; }
        nav a { color: white; text-decoration: none; margin-left: 16px; font-size: 0.95rem; }
        nav a:hover { text-decoration: underline; }
        .container { max-width: 900px; margin: 40px auto; padding: 0 16px; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 32px; margin-bottom: 24px; }
        h2 { margin-bottom: 20px; font-size: 1.2rem; color: #4a90e2; }
        .upload-area { border: 2px dashed #4a90e2; border-radius: 8px; padding: 40px; text-align: center; background: #f7fbff; }
        .upload-area input[type=file] { margin: 16px 0; display: block; width: 100%; }
        .btn { display: inline-block; padding: 10px 24px; background: #4a90e2; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.95rem; text-decoration: none; }
        .btn:hover { background: #357abd; }
        .btn-danger { background: #e24a4a; }
        .btn-danger:hover { background: #bd3535; }
        .file-list { list-style: none; }
        .file-item { display: flex; align-items: center; padding: 12px 16px; border-bottom: 1px solid #eee; gap: 12px; }
        .file-item:last-child { border-bottom: none; }
        .file-icon { font-size: 1.5rem; }
        .file-info { flex: 1; }
        .file-name { font-weight: bold; word-break: break-all; }
        .file-size { font-size: 0.8rem; color: #888; margin-top: 2px; }
        .file-actions a { margin-left: 8px; font-size: 0.85rem; }
        .alert { padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; }
        .alert-success { background: #d4edda; color: #155724; }
        .alert-error { background: #f8d7da; color: #721c24; }
        .empty { text-align: center; color: #999; padding: 32px; }
        .preview-img { max-width: 100%; max-height: 400px; display: block; margin: 0 auto; border-radius: 6px; }
    </style>
</head>
<body>
    <header>
        <h1>&#128196; File Uploader</h1>
        <nav>
            <a href="{{ url_for('index') }}">Upload</a>
            <a href="{{ url_for('file_list') }}">My Files</a>
        </nav>
    </header>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

UPLOAD_PAGE = BASE_HTML.replace("{% block content %}{% endblock %}", """
{% block content %}
{% if message %}
<div class="alert alert-{{ msg_type }}">{{ message }}</div>
{% endif %}
<div class="card">
    <h2>&#8679; Upload a File</h2>
    <form method="POST" action="{{ url_for('upload_file') }}" enctype="multipart/form-data">
        <div class="upload-area">
            <p style="font-size:1.1rem; color:#555;">Select a document or image to upload</p>
            <input type="file" name="file" required>
            <button type="submit" class="btn">Upload File</button>
        </div>
    </form>
</div>
<div style="color:#888; font-size:0.85rem; text-align:center;">
    Allowed types: PNG, JPG, GIF, PDF, TXT, DOC, DOCX, XLS, XLSX, PPT, PPTX, CSV, ZIP, MP4, MP3, BMP, WEBP<br>
    Max size: 50 MB
</div>
{% endblock %}
""")

LIST_PAGE = BASE_HTML.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="card">
    <h2>&#128193; Uploaded Files ({{ files|length }})</h2>
    {% if files %}
    <ul class="file-list">
        {% for f in files %}
        <li class="file-item">
            <span class="file-icon">{{ f.icon }}</span>
            <div class="file-info">
                <div class="file-name">{{ f.original_name }}</div>
                <div class="file-size">{{ f.size }}</div>
            </div>
            <div class="file-actions">
                <a href="{{ url_for('view_file', filename=f.stored_name) }}" class="btn" target="_blank">View</a>
                <a href="{{ url_for('download_file', filename=f.stored_name) }}" class="btn" style="background:#6c757d;">Download</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="empty">
        <p>No files uploaded yet.</p>
        <br>
        <a href="{{ url_for('index') }}" class="btn">Upload your first file</a>
    </div>
    {% endif %}
</div>
{% endblock %}
""")

VIEW_PAGE = BASE_HTML.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="card">
    <h2>&#128065; {{ original_name }}</h2>
    {% if is_image %}
    <img class="preview-img" src="{{ url_for('serve_file', filename=filename) }}" alt="{{ original_name }}">
    {% elif is_text %}
    <pre style="background:#f8f9fa; padding:16px; border-radius:6px; overflow:auto; max-height:500px; white-space:pre-wrap; word-break:break-all;">{{ text_content }}</pre>
    {% elif is_pdf %}
    <iframe src="{{ url_for('serve_file', filename=filename) }}" style="width:100%; height:600px; border:none; border-radius:6px;"></iframe>
    {% else %}
    <div class="empty">
        <p>Preview not available for this file type.</p>
    </div>
    {% endif %}
    <div style="margin-top:20px; display:flex; gap:12px;">
        <a href="{{ url_for('download_file', filename=filename) }}" class="btn">Download</a>
        <a href="{{ url_for('file_list') }}" class="btn" style="background:#6c757d;">Back to List</a>
    </div>
</div>
{% endblock %}
""")

file_registry = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_icon(extension):
    ext = extension.lower()
    if ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}:
        return '🖼️'
    elif ext == 'pdf':
        return '📄'
    elif ext in {'doc', 'docx'}:
        return '📝'
    elif ext in {'xls', 'xlsx', 'csv'}:
        return '📊'
    elif ext in {'ppt', 'pptx'}:
        return '📑'
    elif ext == 'txt':
        return '📃'
    elif ext == 'zip':
        return '🗜️'
    elif ext in {'mp4', 'mp3'}:
        return '🎵'
    return '📁'

def format_size(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"

@app.route('/', methods=['GET'])
def index():
    return render_template_string(UPLOAD_PAGE, message=None, msg_type=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template_string(UPLOAD_PAGE, message="No file part in request.", msg_type="error")

    file = request.files['file']

    if file.filename == '':
        return render_template_string(UPLOAD_PAGE, message="No file selected.", msg_type="error")

    if not allowed_file(file.filename):
        return render_template_string(UPLOAD_PAGE, message="File type not allowed.", msg_type="error")

    original_name = file.filename
    extension = original_name.rsplit('.', 1)[1].lower() if '.' in original_name else ''
    unique_name = f"{uuid.uuid4().hex}.{extension}" if extension else uuid.uuid4().hex

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(save_path)

    size_bytes = os.path.getsize(save_path)

    file_registry[unique_name] = {
        'original_name': original_name,
        'stored_name': unique_name,
        'extension': extension,
        'size': format_size(size_bytes),
        'icon': get_file_icon(extension)
    }

    return render_template_string(
        UPLOAD_PAGE,
        message=f"File '{original_name}' uploaded successfully!",
        msg_type="success"
    )

@app.route('/files', methods=['GET'])
def file_list():
    files = list(file_registry.values())
    return render_template_string(LIST_PAGE, files=files)

@app.route('/view/<filename>', methods=['GET'])
def view_file(filename):
    if filename not in file_registry:
        abort(404)

    info = file_registry[filename]
    ext = info['extension'].lower()
    is_image = ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    is_text = ext in {'txt', 'csv'}
    is_pdf = ext == 'pdf'

    text_content = None
    if is_text:
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text_content = f.read()
        except Exception:
            text_content = "Unable to read file content."

    return render_template_string(
        VIEW_PAGE,
        filename=filename,
        original_name=info['original_name'],
        is_image=is_image,
        is_text=is_text,
        is_pdf=is_pdf,
        text_content=text_content
    )

@app.route('/serve/<filename>', methods=['GET'])
def serve_file(filename):
    if filename not in file_registry:
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    if filename not in file_registry:
        abort(404)
    original_name = file_registry[filename]['original_name']
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        as_attachment=True,
        download_name=original_name
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)