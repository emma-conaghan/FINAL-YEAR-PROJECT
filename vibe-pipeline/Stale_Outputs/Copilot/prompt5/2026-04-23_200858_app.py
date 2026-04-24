from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

UPLOAD_PAGE = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload Documents or Images</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p><a href="{{ url_for('files') }}">View Uploaded Files</a></p>
'''

FILES_PAGE = '''
<!doctype html>
<title>Files</title>
<h1>Uploaded Files</h1>
<ul>
{% for filename in files %}
  <li>
    {{ filename }}
    - <a href="{{ url_for('view_file', filename=filename) }}">View/Open</a>
    - <a href="{{ url_for('download_file', filename=filename) }}">Download</a>
  </li>
{% endfor %}
</ul>
<p><a href="{{ url_for('upload') }}">Upload More Files</a></p>
'''

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            fname = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
            return redirect(url_for('files'))
    return render_template_string(UPLOAD_PAGE)

@app.route('/files')
def files():
    all_files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(FILES_PAGE, files=all_files)

@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)