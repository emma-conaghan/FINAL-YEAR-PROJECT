import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>File Manager</title>
</head>
<body>
    <h1>Upload a Document or Image</h1>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file_to_save">
        <input type="submit" value="Upload">
    </form>
    <hr>
    <h1>File List</h1>
    <ul>
        {% for filename in files %}
        <li>
            <a href="{{ url_for('view_file', filename=filename) }}">{{ filename }}</a>
        </li>
        {% endfor %}
    </ul>
</body>
</html>
"""

@app.route('/')
def index():
    files = os.listdir(UPLOAD_FOLDER)
    return render_template_string(INDEX_HTML, files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file_to_save' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file_to_save']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return redirect(url_for('index'))

@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)