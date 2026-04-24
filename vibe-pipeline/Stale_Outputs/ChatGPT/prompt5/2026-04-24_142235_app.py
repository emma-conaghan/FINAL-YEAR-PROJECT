from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'}

UPLOAD_PAGE = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file><br><br>
  <input type=submit value=Upload>
</form>
<br>
<a href="{{ url_for('files_list') }}">See uploaded files</a>
'''

FILES_LIST_PAGE = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
  {% for file in files %}
    <li>
      <a href="{{ url_for('serve_file', filename=file) }}">{{ file }}</a> 
      - <a href="{{ url_for('serve_file', filename=file) }}" download>Download</a>
    </li>
  {% else %}
    <li>No files uploaded yet.</li>
  {% endfor %}
</ul>
<br>
<a href="{{ url_for('upload') }}">Upload new file</a>
'''

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('files_list'))
    return render_template_string(UPLOAD_PAGE)

@app.route('/files')
def files_list():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files = [f for f in files if allowed_file(f)]
    return render_template_string(FILES_LIST_PAGE, files=files)

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)