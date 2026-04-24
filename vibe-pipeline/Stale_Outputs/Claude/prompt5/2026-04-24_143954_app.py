import os
import uuid
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, abort

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>File Upload App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .upload-box { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }
        .file-list { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        input[type=file] { margin: 10px 0; }
        input[type=submit] { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        input[type=submit]:hover { background: #45a049; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }
        th { background: #f2f2f2; }
        a { color: #2196F3; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .msg { padding: 10px; background: #dff0d8; border: 1px solid #d6e9c6; border-radius: 4px; margin-bottom: 15px; color: #3c763d; }
        .err { padding: 10px; background: #f2dede; border: 1px solid #ebccd1; border-radius: 4px; margin-bottom: 15px; color: #a94442; }
    </style>
</head>
<body>
    <h1>📁 File Upload App</h1>
    {% if message %}
    <div class="msg">{{ message }}</div>
    {% endif %}
    {% if error %}
    <div class="err">{{ error }}</div>
    {% endif %}
    <div class="upload-box">
        <h2>Upload a File</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required><br>
            <input type="submit" value="Upload">
        </form>
    </div>
    <div class="file-list">
        <h2>Uploaded Files</h2>
        {% if files %}
        <table>
            <tr><th>Filename</th><th>Size</th><th>Actions</th></tr>
            {% for f in files %}
            <tr>
                <td>{{ f.original_name }}</td>
                <td>{{ f.size }}</td>
                <td>
                    <a href="/view/{{ f.stored_name }}" target="_blank">View</a> |
                    <a href="/download/{{ f.stored_name }}">Download</a>
                </td>
            </tr>
            {% endfor %}
        </table>
        {% else %}
        <p>No files uploaded yet.</p>
        {% endif %}
    </div>
</body>
</html>
"""

file_registry = []

def human_readable_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

@app.route('/')
def index():
    message = request.args.get('message', '')
    error = request.args.get('error', '')
    return render_template_string(INDEX_TEMPLATE, files=file_registry, message=message, error=error)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index', error='No file part in the request.'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', error='No file selected.'))
    original_name = file.filename
    ext = os.path.splitext(original_name)[1]
    stored_name = str(uuid.uuid4()) + ext
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(save_path)
    size = os.path.getsize(save_path)
    file_registry.append({
        'original_name': original_name,
        'stored_name': stored_name,
        'size': human_readable_size(size)
    })
    return redirect(url_for('index', message=f'File "{original_name}" uploaded successfully.'))

@app.route('/view/<stored_name>')
def view_file(stored_name):
    safe_name = os.path.basename(stored_name)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.exists(file_path):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name)

@app.route('/download/<stored_name>')
def download_file(stored_name):
    safe_name = os.path.basename(stored_name)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.exists(file_path):
        abort(404)
    original_name = safe_name
    for entry in file_registry:
        if entry['stored_name'] == safe_name:
            original_name = entry['original_name']
            break
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name, as_attachment=True, download_name=original_name)

if __name__ == '__main__':
    app.run(debug=True)