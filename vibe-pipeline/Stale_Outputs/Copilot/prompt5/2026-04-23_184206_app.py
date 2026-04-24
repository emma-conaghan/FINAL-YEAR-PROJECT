from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Simple templates as strings
upload_page = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<br>
<a href="{{ url_for('files') }}">View Uploaded Files</a>
'''

files_page = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
{% for file in files %}
  <li>
    {{ file }}
    -
    <a href="{{ url_for('open_file', filename=file) }}">View/Open</a>
    -
    <a href="{{ url_for('download_file', filename=file) }}">Download</a>
  </li>
{% endfor %}
</ul>
<br>
<a href="{{ url_for('upload_file') }}">Upload More Files</a>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('files'))
    return render_template_string(upload_page)

@app.route('/files')
def files():
    file_list = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(files_page, files=file_list)

@app.route('/open/<filename>')
def open_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)