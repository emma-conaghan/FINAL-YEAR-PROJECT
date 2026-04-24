from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

INDEX_HTML = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload Document or Image</h1>
<form method=post enctype=multipart/form-data action="{{ url_for('upload') }}">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<a href="{{ url_for('files') }}">View uploaded files</a>
'''

LIST_HTML = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
{% for filename in files %}
  <li>
    {{ filename }} -
    <a href="{{ url_for('view_file', filename=filename) }}">View</a> |
    <a href="{{ url_for('download_file', filename=filename) }}">Download</a>
  </li>
{% else %}
  <li>No files uploaded.</li>
{% endfor %}
</ul>
<a href="{{ url_for('index') }}">Back to upload</a>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file and file.filename:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('files'))

@app.route('/files')
def files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(LIST_HTML, files=files)

@app.route('/files/view/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/files/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)