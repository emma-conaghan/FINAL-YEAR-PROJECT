from flask import Flask, request, redirect, url_for, render_template_string
import json
import os
import uuid
from datetime import datetime

app = Flask(__name__)

NOTES_FILE = "notes.json"

def load_notes():
    if not os.path.exists(NOTES_FILE):
        return {}
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
        body { font-family: Arial, sans-serif; background: #f5f5f5; color: #333; }
        header {
            background: #4a90d9;
            color: white;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 1.5rem; }
        header a { color: white; text-decoration: none; background: #357abd; padding: 8px 14px; border-radius: 4px; font-size: 0.9rem; }
        header a:hover { background: #2868a0; }
        main { max-width: 900px; margin: 30px auto; padding: 0 16px; }
        .note-card {
            background: white;
            border-radius: 6px;
            padding: 18px 22px;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }
        .note-card h2 { font-size: 1.2rem; margin-bottom: 6px; color: #222; }
        .note-card p { color: #555; font-size: 0.95rem; white-space: pre-wrap; word-break: break-word; }
        .note-meta { font-size: 0.78rem; color: #888; margin-bottom: 10px; }
        .note-actions { margin-top: 12px; display: flex; gap: 10px; }
        .btn {
            padding: 7px 14px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            text-decoration: none;
            display: inline-block;
        }
        .btn-edit { background: #f0ad4e; color: white; }
        .btn-edit:hover { background: #d9930a; }
        .btn-delete { background: #d9534f; color: white; }
        .btn-delete:hover { background: #b52b27; }
        .btn-primary { background: #4a90d9; color: white; }
        .btn-primary:hover { background: #357abd; }
        .btn-secondary { background: #aaa; color: white; }
        .btn-secondary:hover { background: #888; }
        form { background: white; border-radius: 6px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 6px; font-weight: bold; font-size: 0.95rem; }
        form input[type=text], form textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 1rem;
            margin-bottom: 16px;
            font-family: inherit;
        }
        form textarea { height: 200px; resize: vertical; }
        .form-actions { display: flex; gap: 10px; }
        .empty-msg { text-align: center; color: #888; margin-top: 60px; font-size: 1.1rem; }
        .flash { background: #dff0d8; border: 1px solid #d6e9c6; color: #3c763d; padding: 12px 18px; border-radius: 4px; margin-bottom: 16px; }
    </style>
</head>
<body>
    <header>
        <h1>📝 Personal Notes</h1>
        <a href="{{ url_for('new_note') }}">+ New Note</a>
    </header>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
"""

INDEX_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block content %}{% endblock %}",
    """
{% block content %}
{% if notes %}
    {% for note_id, note in notes.items() %}
    <div class="note-card">
        <h2>{{ note.title }}</h2>
        <div class="note-meta">Created: {{ note.created }} {% if note.updated %} &nbsp;|&nbsp; Updated: {{ note.updated }}{% endif %}</div>
        <p>{{ note.content[:300] }}{% if note.content|length > 300 %}...{% endif %}</p>
        <div class="note-actions">
            <a href="{{ url_for('view_note', note_id=note_id) }}" class="btn btn-primary">View</a>
            <a href="{{ url_for('edit_note', note_id=note_id) }}" class="btn btn-edit">Edit</a>
            <form method="POST" action="{{ url_for('delete_note', note_id=note_id) }}" style="display:inline;" onsubmit="return confirm('Delete this note?');">
                <button type="submit" class="btn btn-delete">Delete</button>
            </form>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="empty-msg">No notes yet. <a href="{{ url_for('new_note') }}">Create your first note!</a></div>
{% endif %}
{% endblock %}
"""
)

VIEW_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block content %}{% endblock %}",
    """
{% block content %}
<div class="note-card">
    <h2>{{ note.title }}</h2>
    <div class="note-meta">Created: {{ note.created }} {% if note.updated %} &nbsp;|&nbsp; Updated: {{ note.updated }}{% endif %}</div>
    <p>{{ note.content }}</p>
    <div class="note-actions">
        <a href="{{ url_for('edit_note', note_id=note_id) }}" class="btn btn-edit">Edit</a>
        <a href="{{ url_for('index') }}" class="btn btn-secondary">Back</a>
    </div>
</div>
{% endblock %}
"""
)

FORM_TEMPLATE = BASE_TEMPLATE.replace(
    "{% block content %}{% endblock %}",
    """
{% block content %}
<h2 style="margin-bottom:18px;">{{ form_title }}</h2>
<form method="POST">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" value="{{ note.title if note else '' }}" placeholder="Enter note title..." required>
    <label for="content">Content</label>
    <textarea id="content" name="content" placeholder="Write your note here...">{{ note.content if note else '' }}</textarea>
    <div class="form-actions">
        <button type="submit" class="btn btn-primary">Save Note</button>
        <a href="{{ url_for('index') }}" class="btn btn-secondary">Cancel</a>
    </div>
</form>
{% endblock %}
"""
)

@app.route("/")
def index():
    notes = load_notes()
    sorted_notes = dict(
        sorted(notes.items(), key=lambda x: x[1].get("created", ""), reverse=True)
    )
    return render_template_string(INDEX_TEMPLATE, notes=sorted_notes)

@app.route("/note/new", methods=["GET", "POST"])
def new_note():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if not title:
            title = "Untitled"
        notes = load_notes()
        note_id = str(uuid.uuid4())
        notes[note_id] = {
            "title": title,
            "content": content,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "updated": None,
        }
        save_notes(notes)
        return redirect(url_for("view_note", note_id=note_id))
    return render_template_string(FORM_TEMPLATE, form_title="New Note", note=None)

@app.route("/note/<note_id>")
def view_note(note_id):
    notes = load_notes()
    note = notes.get(note_id)
    if note is None:
        return redirect(url_for("index"))
    return render_template_string(VIEW_TEMPLATE, note=note, note_id=note_id)

@app.route("/note/<note_id>/edit", methods=["GET", "POST"])
def edit_note(note_id):
    notes = load_notes()
    note = notes.get(note_id)
    if note is None:
        return redirect(url_for("index"))
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if not title:
            title = "Untitled"
        notes[note_id]["title"] = title
        notes[note_id]["content"] = content
        notes[note_id]["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_notes(notes)
        return redirect(url_for("view_note", note_id=note_id))
    return render_template_string(FORM_TEMPLATE, form_title="Edit Note", note=note)

@app.route("/note/<note_id>/delete", methods=["POST"])
def delete_note(note_id):
    notes = load_notes()
    if note_id in notes:
        del notes[note_id]
        save_notes(notes)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)