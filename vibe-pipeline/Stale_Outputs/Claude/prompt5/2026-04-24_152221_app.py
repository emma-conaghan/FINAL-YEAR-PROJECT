import os
import mimetypes
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>File Upload App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #333; }
        .upload-form { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        input[type=file] { margin: 10px 0; }
        input[type=submit] { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        input[type=submit]:hover { background: #45a049; }
        .file-list { list-style: none; padding: 0; }
        .file-list li { padding: 10px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }
        .file-list li:hover { background: #f9f9f9; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .message { color: green; font-weight: bold; margin-bottom: 10px; }
        .error { color: red; font-weight: bold; margin-bottom: 10px; }
        .no-files { color: #666; font-style: italic; }
        .actions a { margin-left: 10px; }
    </style>
</head>
<body>
    <h1>File Upload App</h1>
    {% if message %}
        <p class="message">{{ message }}</p>
    {% endif %}
    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}
    <div class="upload-form">
        <h2>Upload a File</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required><br>
            <input type="submit" value="Upload">
        </form>
    </div>
    <h2>Uploaded Files</h2>
    {% if files %}
        <ul class="file-list">
            {% for file in files %}
            <li>
                <span>{{ file }}</span>
                <span class="actions">
                    <a href="/view/{{ file }}" target="_blank">View</a>
                    <a href="/download/{{ file }}">Download</a>
                </span>
            </li>
            {% endfor %}
        </ul>
    {% else %}
        <p class="no-files">No files uploaded yet.</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    message = request.args.get('message', '')
    error = request.args.get('error', '')
    files = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    return render_template_string(INDEX_TEMPLATE, files=files, message=message, error=error)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index', error='No file part in request.'))
    f = request.files['file']
    if f.filename == '':
        return redirect(url_for('index', error='No file selected.'))
    filename = secure_filename(f.filename)
    if not filename:
        return redirect(url_for('index', error='Invalid filename.'))
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)
    return redirect(url_for('index', message=f'File "{filename}" uploaded successfully.'))

@app.route('/view/<filename>')
def view_file(filename):
    filename = secure_filename(filename)
    directory = os.path.abspath(app.config['UPLOAD_FOLDER'])
    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
        return 'File not found.', 404
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    return send_from_directory(directory, filename, mimetype=mime_type)

@app.route('/download/<filename>')
def download_file(filename):
    filename = secure_filename(filename)
    directory = os.path.abspath(app.config['UPLOAD_FOLDER'])
    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
        return 'File not found.', 404
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)