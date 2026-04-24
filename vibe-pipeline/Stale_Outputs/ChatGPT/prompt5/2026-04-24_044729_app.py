from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_HTML = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
  {% for filename in files %}
    <li><a href="{{ url_for('uploaded_file', filename=filename) }}" target="_blank">{{ filename }}</a></li>
  {% else %}
    <li>No files uploaded yet</li>
  {% endfor %}
</ul>
<p><a href="{{ url_for('upload') }}">Upload a new file</a></p>
'''

UPLOAD_HTML = '''
<!doctype html>
<title>Upload new File</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p><a href="{{ url_for('index') }}">Back to file list</a></p>
'''

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(INDEX_HTML, files=files)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('index'))
    return render_template_string(UPLOAD_HTML)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)