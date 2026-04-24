import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>File Upload App</title></head>
<body>
<h1>Upload a File</h1>
<form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <input type="submit" value="Upload">
</form>
<h2>Uploaded Files</h2>
{% if files %}
<ul>
    {% for f in files %}
    <li><a href="/files/{{ f }}" target="_blank">{{ f }}</a></li>
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
    return render_template_string(INDEX_TEMPLATE, files=files)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('index'))

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)