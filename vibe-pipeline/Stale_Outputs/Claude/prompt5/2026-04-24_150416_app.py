import os
import uuid
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>File Manager</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1, h2 { color: #333; }
        .upload-form { background: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        input[type=file] { margin: 10px 0; }
        input[type=submit] { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        input[type=submit]:hover { background: #45a049; }
        .file-list { list-style: none; padding: 0; }
        .file-list li { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .file-list li:hover { background: #f5f5f5; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .message { color: green; font-weight: bold; margin-bottom: 10px; }
        .error { color: red; font-weight: bold; margin-bottom: 10px; }
        .file-name { flex: 1; }
        .file-actions { display: flex; gap: 10px; }
    </style>
</head>
<body>
    <h1>File Manager</h1>

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
                <span class="file-name">{{ file }}</span>
                <span class="file-actions">
                    <a href="/view/{{ file }}" target="_blank">View</a>
                    <a href="/download/{{ file }}">Download</a>
                </span>
            </li>
            {% endfor %}
        </ul>
    {% else %}
        <p>No files uploaded yet.</p>
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
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', error='No file selected.'))
    original_name = file.filename
    safe_name = str(uuid.uuid4()) + '_' + os.path.basename(original_name)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    file.save(save_path)
    return redirect(url_for('index', message=f'File "{original_name}" uploaded successfully as "{safe_name}".'))

@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)