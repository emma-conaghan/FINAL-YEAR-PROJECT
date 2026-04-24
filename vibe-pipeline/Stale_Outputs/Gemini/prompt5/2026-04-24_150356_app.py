import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_DIR = "uploads"

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple File Server</title>
</head>
<body>
    <h1>Upload a Document or Image</h1>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="user_file">
        <input type="submit" value="Upload">
    </form>
    <hr>
    <h2>Stored Files</h2>
    <ul>
        {% for filename in files %}
            <li>
                <a href="{{ url_for('download_file', name=filename) }}">{{ filename }}</a>
            </li>
        {% endfor %}
    </ul>
</body>
</html>
"""

@app.route('/')
def index():
    files = os.listdir(UPLOAD_DIR)
    return render_template_string(TEMPLATE, files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    file_obj = request.files.get('user_file')
    if file_obj and file_obj.filename != '':
        file_obj.save(os.path.join(UPLOAD_DIR, file_obj.filename))
    return redirect(url_for('index'))

@app.route('/view/<name>')
def download_file(name):
    return send_from_directory(UPLOAD_DIR, name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)