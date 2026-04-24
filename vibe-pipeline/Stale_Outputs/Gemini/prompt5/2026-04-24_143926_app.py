import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple File Manager</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6; }
        .upload-section { background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .file-list { list-style: none; padding: 0; }
        .file-item { padding: 10px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Document & Image Server</h1>
    
    <div class="upload-section">
        <h3>Upload File</h3>
        <form method="post" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file_data" required>
            <input type="submit" value="Upload">
        </form>
    </div>

    <h2>Uploaded Files</h2>
    <ul class="file-list">
        {% for filename in files %}
            <li class="file-item">
                <span>{{ filename }}</span>
                <a href="{{ url_for('serve_file', filename=filename) }}" target="_blank">View / Download</a>
            </li>
        {% else %}
            <li>No files uploaded yet.</li>
        {% endfor %}
    </ul>
</body>
</html>
"""

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(INDEX_HTML, files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file_data' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file_data']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return redirect(url_for('index'))

@app.route('/view/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)