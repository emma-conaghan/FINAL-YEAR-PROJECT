import os
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple File Server</title>
</head>
<body>
    <h1>Upload New Document</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    <hr>
    <h1>Stored Files</h1>
    <ul>
        {% for filename in files %}
            <li>
                <a href="{{ url_for('get_file', filename=filename) }}">{{ filename }}</a>
            </li>
        {% endfor %}
    </ul>
</body>
</html>
"""

@app.route('/')
def index():
    files = os.listdir(UPLOAD_FOLDER)
    return render_template_string(HTML_PAGE, files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file_obj = request.files['file']
    if file_obj.filename == '':
        return redirect(url_for('index'))
    if file_obj:
        save_path = os.path.join(UPLOAD_FOLDER, file_obj.filename)
        file_obj.save(save_path)
    return redirect(url_for('index'))

@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)