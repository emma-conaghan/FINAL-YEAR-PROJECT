from flask import Flask, request, redirect, url_for, render_template_string
import json
import os

app = Flask(__name__)

NOTES_FILE = "notes.json"

def load_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    with open(NOTES_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

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
        body { font-family: Arial, sans-serif; background: #f5f5f5; color: #333; }
        header { background: #4a90d9; color: white; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 1.5rem; }
        header a { color: white; text-decoration: none; background: #357abd; padding: 8px 16px; border-radius: 4px; font-size: 0.9rem; }
        header a:hover { background: #2a6099; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 16px; }
        .note-card { background: white; border-radius: 6px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        .note-card h2 { font-size: 1.2rem; margin-bottom: 8px; color: #222; }
        .note-card p { color: #555; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }
        .note-actions { margin-top: 12px; display: flex; gap: 10px; }
        .btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; text-decoration: none; display: inline-block; }
        .btn-edit { background: #f0ad4e; color: white; }
        .btn-edit:hover { background: #d9953a; }
        .btn-delete { background: #d9534f; color: white; }
        .btn-delete:hover { background: #b52b27; }
        .btn-save { background: #5cb85c; color: white; font-size: 1rem; padding: 10px 24px; }
        .btn-save:hover { background: #449d44; }
        .btn-cancel { background: #aaa; color: white; font-size: 1rem; padding: 10px 24px; }
        .btn-cancel:hover { background: #888; }
        form { background: white; border-radius: 6px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 6px; font-weight: bold; color: #444; }
        form input[type=text], form textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 1rem; margin-bottom: 16px; font-family: Arial, sans-serif; }
        form textarea { min-height: 200px; resize: vertical; }
        form input[type=text]:focus, form textarea:focus { outline: none; border-color: #4a90d9; }
        .form-actions { display: flex; gap: 12px; }
        .empty-msg { text-align: center; color: #888; margin-top: 60px; font-size: 1.1rem; }
        .note-meta { font-size: 0.8rem; color: #999; margin-bottom: 8px; }
        h2.page-title { margin-bottom: 20px; font-size: 1.4rem; color: #333; }
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

INDEX_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block content %}{% endblock %}",
    """{% block content %}
    {% if notes %}
        {% for note in notes %}
        <div class="note-card">
            <div class="note-meta">Note #{{ note.id }}</div>
            <h2>{{ note.title }}</h2>
            <p>{{ note.content }}</p>
            <div class="note-actions">
                <a href="{{ url_for('edit_note', note_id=note.id) }}" class="btn btn-edit">Edit</a>
                <form method="POST" action="{{ url_for('delete_note', note_id=note.id) }}" style="display:inline;" onsubmit="return confirm('Delete this note?');">
                    <button type="submit" class="btn btn-delete">Delete</button>
                </form>
            </div>
        </div>
        {% endfor %}
    {% else %}
        <p class="empty-msg">No notes yet. Click <strong>+ New Note</strong> to get started!</p>
    {% endif %}
{% endblock %}"""
)

FORM_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block content %}{% endblock %}",
    """{% block content %}
    <h2 class="page-title">{{ form_title }}</h2>
    <form method="POST">
        <label for="title">Title</label>
        <input type="text" id="title" name="title" placeholder="Enter note title..." value="{{ note_title }}" required>
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here...">{{ note_content }}</textarea>
        <div class="form-actions">
            <button type="submit" class="btn btn-save">Save Note</button>
            <a href="{{ url_for('index') }}" class="btn btn-cancel">Cancel</a>
        </div>
    </form>
{% endblock %}"""
)

@app.route("/")
def index():
    notes = load_notes()
    return render_template_string(
        INDEX_TEMPLATE,
        notes=notes
    )

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
    return render_template_string(
        FORM_TEMPLATE,
        form_title="New Note",
        note_title="",
        note_content=""
    )

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
    return render_template_string(
        FORM_TEMPLATE,
        form_title="Edit Note",
        note_title=note["title"],
        note_content=note["content"]
    )

@app.route("/delete/<int:note_id>", methods=["POST"])
def delete_note(note_id):
    notes = load_notes()
    notes = [n for n in notes if n["id"] != note_id]
    save_notes(notes)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)