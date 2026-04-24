import os
import mimetypes
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

UPLOAD_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>File Upload App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        input[type=file] { margin: 10px 0; }
        input[type=submit] { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        input[type=submit]:hover { background: #45a049; }
        .file-list { list-style: none; padding: 0; }
        .file-list li { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .file-list li:last-child { border-bottom: none; }
        a { color: #2196F3; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .btn { padding: 5px 10px; border-radius: 4px; }
        .btn-view { background: #2196F3; color: white; margin-right: 5px; }
        .btn-download { background: #FF9800; color: white; }
        .empty { color: #999; font-style: italic; }
        .flash { background: #dff0d8; color: #3c763d; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>📁 File Upload App</h1>
    {% if message %}
    <div class="flash">{{ message }}</div>
    {% endif %}
    <div class="card">
        <h2>Upload a File</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required><br>
            <input type="submit" value="Upload File">
        </form>
    </div>
    <div class="card">
        <h2>Uploaded Files</h2>
        {% if files %}
        <ul class="file-list">
            {% for file in files %}
            <li>
                <span>{{ file }}</span>
                <span>
                    <a class="btn btn-view" href="/view/{{ file }}" target="_blank">View</a>
                    <a class="btn btn-download" href="/download/{{ file }}">Download</a>
                </span>
            </li>
            {% endfor %}
        </ul>
        {% else %}
        <p class="empty">No files uploaded yet.</p>
        {% endif %}
    </div>
</body>
</html>
"""

def get_uploaded_files():
    upload_folder = app.config['UPLOAD_FOLDER']
    files = []
    if os.path.exists(upload_folder):
        files = sorted(os.listdir(upload_folder))
    return files

@app.route('/')
def index():
    files = get_uploaded_files()
    message = request.args.get('message', '')
    return render_template_string(UPLOAD_PAGE, files=files, message=message)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file part in the request.'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index', message='No file selected.'))
    
    if file:
        filename = secure_filename(file.filename)
        if not filename:
            return redirect(url_for('index', message='Invalid filename.'))
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        return redirect(url_for('index', message=f'File "{filename}" uploaded successfully!'))
    
    return redirect(url_for('index', message='Upload failed.'))

@app.route('/view/<filename>')
def view_file(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return "File not found.", 404
    
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        mimetype=mime_type
    )

@app.route('/download/<filename>')
def download_file(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return "File not found.", 404
    
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)