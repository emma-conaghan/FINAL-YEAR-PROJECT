import os
import uuid
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, abort

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

ALLOWED_EXTENSIONS = {
    'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg',
    'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'csv', 'zip', 'mp4', 'mp3'
}

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .upload-box { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .file-list { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
        input[type=file] { margin: 10px 0; }
        button { background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #ddd; }
        th { background: #f0f0f0; font-weight: bold; }
        tr:hover { background: #fafafa; }
        a { color: #1a73e8; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .msg { padding: 10px 15px; border-radius: 4px; margin-bottom: 20px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .empty { color: #999; font-style: italic; }
    </style>
</head>
<body>
    <h1>📁 File Upload App</h1>

    {% if message %}
    <div class="msg {{ msg_type }}">{{ message }}</div>
    {% endif %}

    <div class="upload-box">
        <h2>Upload a File</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required><br>
            <button type="submit">Upload</button>
        </form>
    </div>

    <div class="file-list">
        <h2>Uploaded Files</h2>
        {% if files %}
        <table>
            <thead>
                <tr>
                    <th>File Name</th>
                    <th>Size</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
            {% for f in files %}
                <tr>
                    <td>{{ f.display_name }}</td>
                    <td>{{ f.size }}</td>
                    <td>
                        <a href="/view/{{ f.stored_name }}" target="_blank">View</a> &nbsp;|&nbsp;
                        <a href="/download/{{ f.stored_name }}">Download</a>
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="empty">No files uploaded yet.</p>
        {% endif %}
    </div>
</body>
</html>
"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size(filepath):
    size = os.path.getsize(filepath)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"

def load_file_list():
    files = []
    if not os.path.exists(UPLOAD_FOLDER):
        return files
    for stored_name in sorted(os.listdir(UPLOAD_FOLDER), key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True):
        filepath = os.path.join(UPLOAD_FOLDER, stored_name)
        if os.path.isfile(filepath):
            # Try to recover original display name from stored name
            parts = stored_name.split('_', 1)
            display_name = parts[1] if len(parts) == 2 else stored_name
            files.append({
                'stored_name': stored_name,
                'display_name': display_name,
                'size': get_file_size(filepath)
            })
    return files

@app.route('/')
def index():
    message = request.args.get('message', '')
    msg_type = request.args.get('type', 'success')
    files = load_file_list()
    return render_template_string(INDEX_TEMPLATE, files=files, message=message, msg_type=msg_type)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file part in request.', type='error'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index', message='No file selected.', type='error'))

    if not allowed_file(file.filename):
        return redirect(url_for('index', message=f'File type not allowed. Allowed types: {", ".join(sorted(ALLOWED_EXTENSIONS))}', type='error'))

    original_filename = file.filename
    unique_id = uuid.uuid4().hex[:8]
    stored_name = f"{unique_id}_{original_filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(save_path)

    return redirect(url_for('index', message=f'File "{original_filename}" uploaded successfully!', type='success'))

@app.route('/view/<path:filename>')
def view_file(filename):
    safe_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(safe_path):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<path:filename>')
def download_file(filename):
    safe_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(safe_path):
        abort(404)
    # Get original display name
    parts = filename.split('_', 1)
    download_name = parts[1] if len(parts) == 2 else filename
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True, download_name=download_name)

if __name__ == '__main__':
    app.run(debug=True)