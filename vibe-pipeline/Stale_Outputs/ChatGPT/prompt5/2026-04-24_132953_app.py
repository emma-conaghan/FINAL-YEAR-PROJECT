from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_PAGE = '''
<!doctype html>
<title>Upload Files</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file multiple>
  <input type=submit value=Upload>
</form>
<a href="{{ url_for('files') }}">View uploaded files</a>
'''

FILES_PAGE = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
  {% for filename in files %}
    <li><a href="{{ url_for('uploaded_file', filename=filename) }}" target="_blank">{{ filename }}</a></li>
  {% else %}
    <li>No files uploaded yet.</li>
  {% endfor %}
</ul>
<a href="{{ url_for('upload_file') }}">Upload more files</a>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        uploaded_files = request.files.getlist('file')
        for f in uploaded_files:
            if f.filename == '':
                continue
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('files'))
    return render_template_string(UPLOAD_PAGE)

@app.route('/files')
def files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(FILES_PAGE, files=files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)