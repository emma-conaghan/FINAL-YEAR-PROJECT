from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

upload_page = '''
<!doctype html>
<title>Upload File</title>
<h1>Upload Document or Image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
<p><a href="{{ url_for('list_files') }}">View uploaded files</a></p>
'''

list_page = '''
<!doctype html>
<title>Uploaded Files</title>
<h1>Uploaded Files</h1>
<ul>
{% for filename in files %}
  <li><a href="{{ url_for('uploaded_file', filename=filename) }}">{{ filename }}</a></li>
{% else %}
  <li><em>No files uploaded yet.</em></li>
{% endfor %}
</ul>
<p><a href="{{ url_for('upload') }}">Upload more files</a></p>
'''

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if f and f.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
            f.save(filepath)
        return redirect(url_for('list_files'))
    return render_template_string(upload_page)

@app.route('/files')
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(list_page, files=files)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)