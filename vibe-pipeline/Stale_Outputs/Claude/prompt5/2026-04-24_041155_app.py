import os
import mimetypes
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, abort

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>File Upload App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #333; }
        .upload-form { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .file-list { list-style: none; padding: 0; }
        .file-list li { padding: 10px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }
        .file-list li:hover { background: #f9f9f9; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        input[type=file] { margin: 10px 0; }
        input[type=submit] { background: #0066cc; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
        input[type=submit]:hover { background: #0052a3; }
        .msg { color: green; margin-bottom: 10px; }
        .err { color: red; margin-bottom: 10px; }
        .actions a { margin-left: 10px; }
    </style>
</head>
<body>
    <h1>File Upload App</h1>
    {% if message %}
    <div class="msg">{{ message }}</div>
    {% endif %}
    {% if error %}
    <div class="err">{{ error }}</div>
    {% endif %}
    <div class="upload-form">
        <h2>Upload a File</h2>
        <form method="post" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required><br>
            <input type="submit" value="Upload">
        </form>
    </div>
    <h2>Uploaded Files</h2>
    {% if files %}
    <ul class="file-list">
        {% for f in files %}
        <li>
            <span>{{ f }}</span>
            <span class="actions">
                <a href="/view/{{ f }}" target="_blank">View</a>
                <a href="/download/{{ f }}">Download</a>
            </span>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <p>No files uploaded yet.</p>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', '')
    error = request.args.get('error', '')
    files = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    return render_template_string(INDEX_TEMPLATE, files=files, message=message, error=error)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index', error='No file part in the request.'))
    f = request.files['file']
    if f.filename == '':
        return redirect(url_for('index', error='No file selected.'))
    filename = os.path.basename(f.filename)
    if not filename:
        return redirect(url_for('index', error='Invalid filename.'))
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)
    return redirect(url_for('index', message=f'File "{filename}" uploaded successfully.'))

@app.route('/view/<path:filename>')
def view_file(filename):
    filename = os.path.basename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(filepath):
        abort(404)
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype=mime_type)

@app.route('/download/<path:filename>')
def download_file(filename):
    filename = os.path.basename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(filepath):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)