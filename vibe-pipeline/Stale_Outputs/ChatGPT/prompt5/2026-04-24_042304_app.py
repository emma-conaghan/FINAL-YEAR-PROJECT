from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'xls', 'xlsx'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

upload_page = '''
<!doctype html>
<title>Upload Document or Image</title>
<h1>Upload Document or Image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p><a href="{{ url_for('file_list') }}">View uploaded files</a></p>
'''

list_page = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
  {% for filename in files %}
    <li><a href="{{ url_for('uploaded_file', filename=filename) }}">{{ filename }}</a> - 
    <a href="{{ url_for('download_file', filename=filename) }}">Download</a></li>
  {% else %}
    <li>No files uploaded yet.</li>
  {% endfor %}
</ul>
<p><a href="{{ url_for('upload_file') }}">Upload more files</a></p>
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('file_list'))
        else:
            return redirect(request.url)
    return render_template_string(upload_page)

@app.route('/files')
def file_list():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files.sort()
    return render_template_string(list_page, files=files)

@app.route('/files/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)