import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple File Server</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        .file-list { margin-top: 20px; }
        .file-item { margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>Upload Document or Image</h1>
    <form method="post" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    <hr>
    <div class="file-list">
        <h2>Uploaded Files</h2>
        {% if files %}
            <ul>
            {% for f in files %}
                <li class="file-item">
                    <a href="{{ url_for('view_file', filename=f) }}" target="_blank">{{ f }}</a>
                </li>
            {% endfor %}
            </ul>
        {% else %}
            <p>No files uploaded yet.</p>
        {% endif %}
    </div>
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
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return redirect(url_for('index'))

@app.route('/files/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)