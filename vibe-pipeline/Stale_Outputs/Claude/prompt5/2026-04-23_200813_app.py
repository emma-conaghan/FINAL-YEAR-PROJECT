import os
import uuid
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
  .file-list li { background: #fff; border: 1px solid #ddd; margin: 8px 0; padding: 12px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; }
  .file-list a { color: #0066cc; text-decoration: none; margin-left: 10px; }
  .file-list a:hover { text-decoration: underline; }
  .btn { background: #0066cc; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 14px; }
  .btn:hover { background: #0052a3; }
  input[type=file] { margin: 10px 0; }
  .no-files { color: #999; font-style: italic; }
  .flash { background: #d4edda; color: #155724; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
  .error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
</style>
</head>
<body>
  <h1>File Upload App</h1>
  {% if message %}
  <div class="flash">{{ message }}</div>
  {% endif %}
  {% if error %}
  <div class="error">{{ error }}</div>
  {% endif %}
  <div class="upload-form">
    <h2>Upload a File</h2>
    <form method="POST" action="/upload" enctype="multipart/form-data">
      <input type="file" name="file" required><br>
      <button type="submit" class="btn">Upload</button>
    </form>
  </div>
  <h2>Uploaded Files</h2>
  {% if files %}
  <ul class="file-list">
    {% for file in files %}
    <li>
      <span>{{ file.original_name }}</span>
      <div>
        <a href="/view/{{ file.stored_name }}" target="_blank">View</a>
        <a href="/download/{{ file.stored_name }}">Download</a>
      </div>
    </li>
    {% endfor %}
  </ul>
  {% else %}
  <p class="no-files">No files uploaded yet.</p>
  {% endif %}
</body>
</html>
'''

file_registry = []

@app.route('/')
def index():
    message = request.args.get('message', '')
    error = request.args.get('error', '')
    return render_template_string(INDEX_HTML, files=file_registry, message=message, error=error)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index', error='No file part in request.'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index', error='No file selected.'))
    
    original_name = file.filename
    ext = os.path.splitext(original_name)[1]
    stored_name = str(uuid.uuid4()) + ext
    
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_name)
    file.save(save_path)
    
    file_registry.append({
        'original_name': original_name,
        'stored_name': stored_name
    })
    
    return redirect(url_for('index', message=f'File "{original_name}" uploaded successfully.'))

@app.route('/view/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
def download_file(filename):
    original_name = filename
    for f in file_registry:
        if f['stored_name'] == filename:
            original_name = f['original_name']
            break
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True, download_name=original_name)

if __name__ == '__main__':
    app.run(debug=True)