import os
import uuid
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>File Upload App</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
  h1 { color: #333; }
  .upload-form { background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
  .file-list { list-style: none; padding: 0; }
  .file-list li { background: #fff; border: 1px solid #ddd; margin: 8px 0; padding: 12px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; }
  .file-list a { color: #0066cc; text-decoration: none; margin-left: 10px; }
  .file-list a:hover { text-decoration: underline; }
  input[type=file] { margin: 10px 0; }
  input[type=submit] { background: #0066cc; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
  input[type=submit]:hover { background: #0052a3; }
  .msg { color: green; font-weight: bold; margin-bottom: 15px; }
  .empty { color: #888; }
</style>
</head>
<body>
<h1>File Upload App</h1>
{% if message %}
<div class="msg">{{ message }}</div>
{% endif %}
<div class="upload-form">
  <h2>Upload a File</h2>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file" required><br>
    <input type="submit" value="Upload">
  </form>
</div>
<h2>Uploaded Files</h2>
{% if files %}
<ul class="file-list">
  {% for filename, original_name in files %}
  <li>
    <span>{{ original_name }}</span>
    <span>
      <a href="/view/{{ filename }}" target="_blank">View</a>
      <a href="/download/{{ filename }}">Download</a>
    </span>
  </li>
  {% endfor %}
</ul>
{% else %}
<p class="empty">No files uploaded yet.</p>
{% endif %}
</body>
</html>
'''

def load_file_index():
    index_path = os.path.join(app.config['UPLOAD_FOLDER'], 'index.txt')
    files = []
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        files.append((parts[0], parts[1]))
    return files

def save_file_to_index(stored_name, original_name):
    index_path = os.path.join(app.config['UPLOAD_FOLDER'], 'index.txt')
    with open(index_path, 'a') as f:
        f.write(f'{stored_name}|{original_name}\n')

@app.route('/')
def index():
    message = request.args.get('message', '')
    files = load_file_index()
    return render_template_string(INDEX_TEMPLATE, files=files, message=message)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index', message='No file part in request.'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', message='No file selected.'))
    original_name = file.filename
    ext = os.path.splitext(original_name)[1]
    stored_name = str(uuid.uuid4()) + ext
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(save_path)
    save_file_to_index(stored_name, original_name)
    return redirect(url_for('index', message=f'File "{original_name}" uploaded successfully!'))

@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    files = load_file_index()
    original_name = filename
    for stored, orig in files:
        if stored == filename:
            original_name = orig
            break
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True, download_name=original_name)

if __name__ == '__main__':
    app.run(debug=True)