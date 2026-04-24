import os
import uuid
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

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
  .upload-form { background: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
  input[type=file] { margin: 10px 0; }
  button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
  button:hover { background: #45a049; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }
  th { background: #4CAF50; color: white; }
  tr:hover { background: #f5f5f5; }
  a { color: #4CAF50; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .no-files { color: #888; font-style: italic; }
</style>
</head>
<body>
<h1>File Upload App</h1>
<div class="upload-form">
  <h2>Upload a File</h2>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file" required><br>
    <button type="submit">Upload</button>
  </form>
</div>
<h2>Uploaded Files</h2>
{% if files %}
<table>
  <tr><th>Filename</th><th>Size</th><th>Actions</th></tr>
  {% for f in files %}
  <tr>
    <td>{{ f.display_name }}</td>
    <td>{{ f.size }}</td>
    <td>
      <a href="/view/{{ f.stored_name }}" target="_blank">View</a> |
      <a href="/download/{{ f.stored_name }}">Download</a> |
      <a href="/delete/{{ f.stored_name }}" onclick="return confirm('Delete this file?')">Delete</a>
    </td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p class="no-files">No files uploaded yet.</p>
{% endif %}
</body>
</html>
'''

def get_human_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def get_files():
    files = []
    folder = app.config['UPLOAD_FOLDER']
    for stored_name in os.listdir(folder):
        filepath = os.path.join(folder, stored_name)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            parts = stored_name.split('_', 1)
            display_name = parts[1] if len(parts) == 2 else stored_name
            files.append({
                'stored_name': stored_name,
                'display_name': display_name,
                'size': get_human_size(size),
                'mtime': os.path.getmtime(filepath)
            })
    files.sort(key=lambda x: x['mtime'], reverse=True)
    return files

@app.route('/')
def index():
    files = get_files()
    return render_template_string(INDEX_HTML, files=files)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    unique_id = uuid.uuid4().hex[:8]
    stored_name = f"{unique_id}_{file.filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(save_path)
    return redirect(url_for('index'))

@app.route('/view/<path:filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/delete/<path:filename>')
def delete_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.isfile(filepath):
        os.remove(filepath)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)