import os
import uuid
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head><title>File Upload</title></head>
<body>
<h1>Upload a File</h1>
<form method="post" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <input type="submit" value="Upload">
</form>
<h2>Uploaded Files</h2>
{% if files %}
<ul>
{% for f in files %}
    <li><a href="/files/{{ f }}" target="_blank">{{ f }}</a> | <a href="/download/{{ f }}">Download</a></li>
{% endfor %}
</ul>
{% else %}
<p>No files uploaded yet.</p>
{% endif %}
</body>
</html>
'''

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files = sorted(files)
    return render_template_string(INDEX_HTML, files=files)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    f = request.files['file']
    if f.filename == '':
        return 'No selected file', 400
    original_name = f.filename
    safe_name = str(uuid.uuid4()) + '_' + os.path.basename(original_name)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    f.save(save_path)
    return redirect(url_for('index'))

@app.route('/files/<path:filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)