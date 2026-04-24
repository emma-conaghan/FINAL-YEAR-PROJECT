import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Simple File Server</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        ul { list-style-type: none; padding: 0; }
        li { margin: 10px 0; border: 1px solid #ddd; padding: 10px; border-radius: 4px; }
        a { text-decoration: none; color: #007bff; }
        .upload-section { margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>File Manager</h1>
    
    <div class="upload-section">
        <h2>Upload New File</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <button type="submit">Upload</button>
        </form>
    </div>

    <hr>

    <h2>Uploaded Files</h2>
    <ul>
        {% for filename in files %}
            <li>
                <a href="{{ url_for('view_file', filename=filename) }}" target="_blank">{{ filename }}</a>
            </li>
        {% endfor %}
    </ul>
    {% if not files %}
        <p>No files uploaded yet.</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template_string(HTML_TEMPLATE, files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('index'))

@app.route('/files/<filename>')
def view_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)