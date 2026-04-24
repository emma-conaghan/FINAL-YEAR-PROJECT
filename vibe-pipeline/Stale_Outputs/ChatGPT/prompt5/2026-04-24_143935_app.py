from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_PAGE_HTML = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p><a href="{{ url_for('list_files') }}">See uploaded files</a></p>
'''

LIST_FILES_HTML = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
{% for filename in files %}
  <li><a href="{{ url_for('uploaded_file', filename=filename) }}">{{ filename }}</a></li>
{% else %}
  <li>No files uploaded yet.</li>
{% endfor %}
</ul>
<p><a href="{{ url_for('upload_file') }}">Upload a new file</a></p>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('list_files'))
    return render_template_string(UPLOAD_PAGE_HTML)

@app.route('/files')
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(LIST_FILES_HTML, files=files)

@app.route('/files/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)