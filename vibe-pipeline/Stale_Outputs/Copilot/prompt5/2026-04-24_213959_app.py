from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'pdf', 'doc', 'docx', 'txt', 'rtf', 'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_PAGE = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload Document or Image</h1>
<form method=post enctype=multipart/form-data action="{{ url_for('upload') }}">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<br>
<a href="{{ url_for('files') }}">View uploaded files</a>
'''

FILES_PAGE = '''
<!doctype html>
<title>Files</title>
<h1>Uploaded Files</h1>
<ul>
{% for f in files %}
  <li>
    {{ f }}
    [<a href="{{ url_for('view_file', filename=f) }}">View</a>]
    [<a href="{{ url_for('download_file', filename=f) }}">Download</a>]
  </li>
{% else %}
  <li>No files uploaded.</li>
{% endfor %}
</ul>
<br>
<a href="{{ url_for('upload_page') }}">Back to upload</a>
'''

@app.route('/')
def upload_page():
    return render_template_string(UPLOAD_PAGE)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('upload_page'))
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
    return redirect(url_for('files'))

@app.route('/files')
def files():
    filelist = os.listdir(UPLOAD_FOLDER)
    return render_template_string(FILES_PAGE, files=filelist)

@app.route('/view/<filename>')
def view_file(filename):
    if '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
        return '''
        <!doctype html>
        <title>View File</title>
        <h1>Viewing: {}</h1>
        <img src="{}" style="max-width:700px;">
        <br><a href="{}">Back to file list</a>
        '''.format(filename, url_for('download_file', filename=filename), url_for('files'))
    else:
        return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)