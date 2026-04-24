from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx'])

upload_page_html = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<a href="{{ url_for('list_files') }}">See uploaded files</a>
'''

list_files_html = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>List of Uploaded Files</h1>
<ul>
{% for filename in files %}
  <li><a href="{{ url_for('uploaded_file', filename=filename) }}">{{ filename }}</a></li>
{% else %}
  <li>No files uploaded yet</li>
{% endfor %}
</ul>
<a href="{{ url_for('upload_file') }}">Upload more files</a>
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            return redirect(url_for('list_files'))
        else:
            return redirect(request.url)
    return render_template_string(upload_page_html)

@app.route('/files')
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(list_files_html, files=files)

@app.route('/files/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)