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
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<br>
<a href="{{ url_for('files') }}">View uploaded files</a>
'''

FILES_PAGE = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
  {% for file in files %}
    <li><a href="{{ url_for('uploaded_file', filename=file) }}">{{ file }}</a></li>
  {% endfor %}
</ul>
<br>
<a href="{{ url_for('upload') }}">Upload a new file</a>
'''

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('files'))
    return render_template_string(UPLOAD_PAGE)

@app.route('/files')
def files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(FILES_PAGE, files=files)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)