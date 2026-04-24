from flask import Flask, request, redirect, url_for, render_template_string
import os
import json

app = Flask(__name__)

NOTES_FILE = "notes.json"

def load_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    with open(NOTES_FILE, "r") as f:
        return json.load(f)

def save_notes(notes):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personal Notes</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f4f4; color: #333; }
        header { background: #4a90e2; color: white; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 1.5rem; }
        header a { color: white; text-decoration: none; background: #357abd; padding: 8px 14px; border-radius: 4px; font-size: 0.9rem; }
        header a:hover { background: #2a6099; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 16px; }
        .note-card { background: white; border-radius: 6px; padding: 18px 22px; margin-bottom: 18px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .note-card h2 { font-size: 1.2rem; margin-bottom: 8px; color: #4a90e2; }
        .note-card p { font-size: 0.95rem; white-space: pre-wrap; line-height: 1.5; color: #555; }
        .note-card .actions { margin-top: 12px; display: flex; gap: 10px; }
        .btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; text-decoration: none; display: inline-block; }
        .btn-edit { background: #4a90e2; color: white; }
        .btn-edit:hover { background: #357abd; }
        .btn-delete { background: #e24a4a; color: white; }
        .btn-delete:hover { background: #bd3535; }
        form { background: white; border-radius: 6px; padding: 24px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 6px; font-weight: bold; font-size: 0.9rem; }
        form input[type="text"], form textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.95rem; margin-bottom: 16px; }
        form textarea { height: 200px; resize: vertical; font-family: Arial, sans-serif; }
        form input[type="submit"] { background: #4a90e2; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 1rem; }
        form input[type="submit"]:hover { background: #357abd; }
        .empty { text-align: center; color: #999; font-size: 1rem; margin-top: 40px; }
        .note-meta { font-size: 0.78rem; color: #999; margin-bottom: 6px; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #4a90e2; text-decoration: none; font-size: 0.9rem; }
        .back-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
<header>
    <h1>📝 Personal Notes</h1>
    <a href="{{ url_for('new_note') }}">+ New Note</a>
</header>
<div class="container">
    {% block content %}{% endblock %}
</div>
</body>
</html>
"""

INDEX_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% if notes %}
    {% for note in notes %}
    <div class="note-card">
        <div class="note-meta">Note #{{ note.id }}</div>
        <h2>{{ note.title }}</h2>
        <p>{{ note.content }}</p>
        <div class="actions">
            <a class="btn btn-edit" href="{{ url_for('edit_note', note_id=note.id) }}">Edit</a>
            <form method="POST" action="{{ url_for('delete_note', note_id=note.id) }}" style="display:inline; background:none; box-shadow:none; padding:0; margin:0;">
                <button class="btn btn-delete" type="submit" onclick="return confirm('Delete this note?')">Delete</button>
            </form>
        </div>
    </div>
    {% endfor %}
{% else %}
    <p class="empty">No notes yet. Click <strong>+ New Note</strong> to get started!</p>
{% endif %}
""")

FORM_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
<a class="back-link" href="{{ url_for('index') }}">← Back to Notes</a>
<form method="POST">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" placeholder="Enter note title..." value="{{ title }}" required>
    <label for="content">Content</label>
    <textarea id="content" name="content" placeholder="Write your note here...">{{ content }}</textarea>
    <input type="submit" value="{{ submit_label }}">
</form>
""")

@app.route("/")
def index():
    notes = load_notes()
    return render_template_string(INDEX_TEMPLATE, notes=notes)

@app.route("/new", methods=["GET", "POST"])
def new_note():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title:
            notes = load_notes()
            new_id = max((n["id"] for n in notes), default=0) + 1
            notes.append({"id": new_id, "title": title, "content": content})
            save_notes(notes)
        return redirect(url_for("index"))
    return render_template_string(FORM_TEMPLATE, title="", content="", submit_label="Create Note")

@app.route("/edit/<int:note_id>", methods=["GET", "POST"])
def edit_note(note_id):
    notes = load_notes()
    note = next((n for n in notes if n["id"] == note_id), None)
    if note is None:
        return redirect(url_for("index"))
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title:
            note["title"] = title
            note["content"] = content
            save_notes(notes)
        return redirect(url_for("index"))
    return render_template_string(FORM_TEMPLATE, title=note["title"], content=note["content"], submit_label="Save Changes")

@app.route("/delete/<int:note_id>", methods=["POST"])
def delete_note(note_id):
    notes = load_notes()
    notes = [n for n in notes if n["id"] != note_id]
    save_notes(notes)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)