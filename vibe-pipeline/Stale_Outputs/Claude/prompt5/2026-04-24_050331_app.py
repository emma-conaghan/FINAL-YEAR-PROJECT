import os
from flask import Flask, render_template_string, request, redirect, url_for, send_from_directory, flash

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>File Upload App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        h1 { color: #333; }
        .upload-form { background: #f4f4f4; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .file-list { list-style: none; padding: 0; }
        .file-list li { padding: 10px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }
        .file-list li:hover { background: #f9f9f9; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .btn { background: #0066cc; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #0052a3; }
        .flash { background: #d4edda; color: #155724; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
        .empty { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <h1>📁 File Upload App</h1>

    {% for message in get_flashed_messages() %}
    <div class="flash">{{ message }}</div>
    {% endfor %}

    <div class="upload-form">
        <h2>Upload a File</h2>
        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <button type="submit" class="btn">Upload</button>
        </form>
    </div>

    <h2>Uploaded Files</h2>
    {% if files %}
    <ul class="file-list">
        {% for filename in files %}
        <li>
            <span>{{ filename }}</span>
            <div>
                <a href="/view/{{ filename }}" target="_blank">Open</a>
                &nbsp;|&nbsp;
                <a href="/download/{{ filename }}">Download</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <p class="empty">No files uploaded yet.</p>
    {% endif %}
</body>
</html>
"""

@app.route("/")
def index():
    files = sorted(os.listdir(app.config["UPLOAD_FOLDER"]))
    return render_template_string(INDEX_TEMPLATE, files=files)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))
    
    file = request.files["file"]
    
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    
    filename = file.filename
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    flash(f"File '{filename}' uploaded successfully!")
    return redirect(url_for("index"))

@app.route("/view/<path:filename>")
def view_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)