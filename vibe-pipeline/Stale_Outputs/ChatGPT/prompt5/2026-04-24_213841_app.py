from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_PAGE = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p><a href="/files">View uploaded files</a></p>
'''

FILES_PAGE = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
{% for filename in files %}
  <li><a href="{{ url_for('uploaded_file', filename=filename) }}" target="_blank">{{ filename }}</a></li>
{% endfor %}
</ul>
<p><a href="/">Upload more files</a></p>
'''

@app.route('/')
def upload_form():
    return render_template_string(UPLOAD_PAGE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('upload_form'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_form'))
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('files_list'))

@app.route('/files')
def files_list():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(FILES_PAGE, files=files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)