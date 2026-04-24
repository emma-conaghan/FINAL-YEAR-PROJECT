import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HTML_TEMPLATE = """
<!doctype html>
<html>
<head><title>File Manager</title></head>
<body>
    <h1>Upload a Document or Image</h1>
    <form method="post" enctype="multipart/form-data" action="/upload">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    <hr>
    <h1>Uploaded Files</h1>
    <ul>
    {% for filename in file_list %}
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
    file_list = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(HTML_TEMPLATE, file_list=file_list)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    uploaded_item = request.files['file']
    if uploaded_item.filename == '':
        return redirect(url_for('index'))
    if uploaded_item:
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_item.filename)
        uploaded_item.save(save_path)
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)