import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>File Manager</title>
</head>
<body>
    <h1>Upload Document or Image</h1>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file_upload">
        <input type="submit" value="Upload">
    </form>
    <hr>
    <h1>Uploaded Files</h1>
    <ul>
        {% for filename in files %}
            <li>
                <a href="{{ url_for('view_file', name=filename) }}">{{ filename }}</a>
            </li>
        {% endfor %}
    </ul>
</body>
</html>
"""

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(HTML_TEMPLATE, files=files)

@app.route('/upload', methods=['POST'])
def upload_logic():
    if 'file_upload' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file_upload']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return redirect(url_for('index'))

@app.route('/files/<name>')
def view_file(name):
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)