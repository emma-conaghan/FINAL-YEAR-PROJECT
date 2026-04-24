from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

upload_page = '''
<!doctype html>
<title>Upload new File</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data action="{{ url_for('upload') }}">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p><a href="{{ url_for('files') }}">View uploaded files</a></p>
'''

files_page = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
{% for filename in files %}
  <li><a href="{{ url_for('uploaded_file', filename=filename) }}">{{ filename }}</a></li>
{% else %}
  <li>No files uploaded</li>
{% endfor %}
</ul>
<p><a href="{{ url_for('upload_page') }}">Upload more files</a></p>
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_page():
    return render_template_string(upload_page)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('upload_page'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_page'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('files'))
    return redirect(url_for('upload_page'))

@app.route('/files')
def files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(files_page, files=files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)