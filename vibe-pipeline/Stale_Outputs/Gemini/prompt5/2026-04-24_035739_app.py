import os
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>File Storage App</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        li { margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>Upload Document or Image</h1>
    <form method="post" enctype="multipart/form-data" action="/upload">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    <hr>
    <h2>Available Files</h2>
    <ul>
        {% for filename in files %}
        <li>
            <a href="{{ url_for('view_file', filename=filename) }}" target="_blank">{{ filename }}</a>
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
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('index'))

@app.route('/files/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)