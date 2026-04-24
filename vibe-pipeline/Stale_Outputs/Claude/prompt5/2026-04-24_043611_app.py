import os
import uuid
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_TEMPLATE = """
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
.btn { background: #0066cc; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }
.btn:hover { background: #0052a3; }
.no-files { color: #999; font-style: italic; }
</style>
</head>
<body>
<h1>📁 File Upload App</h1>

<div class="upload-form">
  <h2>Upload a File</h2>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file" required style="margin-bottom:10px; display:block;">
    <button type="submit" class="btn">Upload</button>
  </form>
  {% if message %}
  <p style="color: green; margin-top: 10px;">{{ message }}</p>
  {% endif %}
</div>

<h2>Uploaded Files</h2>
{% if files %}
<ul class="file-list">
  {% for fname, original in files %}
  <li>
    <span>{{ original }}</span>
    <span>
      <a href="/view/{{ fname }}" target="_blank">View</a> &nbsp;|&nbsp;
      <a href="/download/{{ fname }}">Download</a>
    </span>
  </li>
  {% endfor %}
</ul>
{% else %}
<p class="no-files">No files uploaded yet.</p>
{% endif %}
</body>
</html>
"""

def get_uploaded_files():
    files = []
    meta_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meta.txt')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        files.append((parts[0], parts[1]))
    return files

def save_file_meta(stored_name, original_name):
    meta_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meta.txt')
    with open(meta_path, 'a') as f:
        f.write(f"{stored_name}|{original_name}\n")

@app.route('/', methods=['GET'])
def index():
    files = get_uploaded_files()
    return render_template_string(INDEX_TEMPLATE, files=files, message=request.args.get('message', ''))

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file selected.'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', message='No file selected.'))
    original_name = file.filename
    ext = os.path.splitext(original_name)[1]
    stored_name = str(uuid.uuid4()) + ext
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(save_path)
    save_file_meta(stored_name, original_name)
    return redirect(url_for('index', message=f'File "{original_name}" uploaded successfully!'))

@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    files = get_uploaded_files()
    original_name = filename
    for stored, original in files:
        if stored == filename:
            original_name = original
            break
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True, download_name=original_name)

if __name__ == '__main__':
    app.run(debug=True)