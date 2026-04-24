import os
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head><title>File Upload App</title>
<style>
body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
h1 { color: #333; }
.upload-form { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
.file-list { list-style: none; padding: 0; }
.file-list li { padding: 10px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }
.file-list li:hover { background: #f9f9f9; }
a { color: #0066cc; text-decoration: none; }
a:hover { text-decoration: underline; }
input[type=submit] { background: #0066cc; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
input[type=submit]:hover { background: #0052a3; }
.no-files { color: #888; font-style: italic; }
</style>
</head>
<body>
<h1>File Upload App</h1>
<div class="upload-form">
  <h2>Upload a File</h2>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file" required>
    <input type="submit" value="Upload">
  </form>
</div>
<h2>Uploaded Files</h2>
{% if files %}
<ul class="file-list">
  {% for f in files %}
  <li>
    <span>{{ f }}</span>
    <span>
      <a href="/view/{{ f }}" target="_blank">View</a> &nbsp;|&nbsp;
      <a href="/download/{{ f }}">Download</a>
    </span>
  </li>
  {% endfor %}
</ul>
{% else %}
<p class="no-files">No files uploaded yet.</p>
{% endif %}
</body>
</html>
'''

@app.route('/')
def index():
    files = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    return render_template_string(INDEX_HTML, files=files)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    f = request.files['file']
    if f.filename == '':
        return redirect(url_for('index'))
    filename = f.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)
    return redirect(url_for('index'))

@app.route('/view/<path:filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)