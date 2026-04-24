from flask import Flask, request, redirect, url_for, render_template_string
import json
import os
import time

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
        body { font-family: Arial, sans-serif; background: #f4f4f4; color: #333; padding: 20px; }
        header { background: #4a90e2; color: white; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 1.5em; }
        header a { color: white; text-decoration: none; background: rgba(255,255,255,0.2); padding: 8px 14px; border-radius: 5px; }
        header a:hover { background: rgba(255,255,255,0.35); }
        .container { max-width: 900px; margin: 0 auto; }
        .note-card { background: white; border-radius: 8px; padding: 18px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.08); }
        .note-card h2 { font-size: 1.2em; margin-bottom: 8px; color: #222; }
        .note-card p.content { color: #555; white-space: pre-wrap; word-break: break-word; }
        .note-card .meta { font-size: 0.8em; color: #999; margin-top: 10px; }
        .note-card .actions { margin-top: 12px; display: flex; gap: 10px; }
        .btn { padding: 7px 14px; border: none; border-radius: 5px; cursor: pointer; font-size: 0.9em; text-decoration: none; display: inline-block; }
        .btn-primary { background: #4a90e2; color: white; }
        .btn-primary:hover { background: #357abd; }
        .btn-danger { background: #e24a4a; color: white; }
        .btn-danger:hover { background: #bd3535; }
        .btn-secondary { background: #888; color: white; }
        .btn-secondary:hover { background: #666; }
        form { background: white; padding: 24px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.08); }
        form label { display: block; margin-bottom: 6px; font-weight: bold; color: #444; }
        form input[type=text], form textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px; font-size: 1em; margin-bottom: 16px; }
        form textarea { height: 200px; resize: vertical; font-family: Arial, sans-serif; }
        .empty { text-align: center; color: #999; padding: 40px; background: white; border-radius: 8px; }
        .flash { padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>📝 Personal Notes</h1>
        <a href="{{ url_for('index') }}">All Notes</a>
    </header>
    {% block content %}{% endblock %}
</div>
</body>
</html>
"""

INDEX_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<div style="margin-bottom:15px; display:flex; justify-content:flex-end;">
    <a href="{{ url_for('new_note') }}" class="btn btn-primary">+ New Note</a>
</div>
{% if notes %}
    {% for note in notes %}
    <div class="note-card">
        <h2>{{ note.title }}</h2>
        <p class="content">{{ note.content[:300] }}{% if note.content|length > 300 %}...{% endif %}</p>
        <div class="meta">Created: {{ note.created_at }} &nbsp;|&nbsp; Last edited: {{ note.updated_at }}</div>
        <div class="actions">
            <a href="{{ url_for('view_note', note_id=note.id) }}" class="btn btn-secondary">View</a>
            <a href="{{ url_for('edit_note', note_id=note.id) }}" class="btn btn-primary">Edit</a>
            <form method="POST" action="{{ url_for('delete_note', note_id=note.id) }}" style="margin:0;padding:0;box-shadow:none;" onsubmit="return confirm('Delete this note?')">
                <button type="submit" class="btn btn-danger">Delete</button>
            </form>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="empty">
        <p style="font-size:1.2em;">No notes yet!</p>
        <p style="margin-top:10px;"><a href="{{ url_for('new_note') }}" class="btn btn-primary">Create your first note</a></p>
    </div>
{% endif %}
{% endblock %}
""")

VIEW_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="note-card">
    <h2 style="font-size:1.5em;">{{ note.title }}</h2>
    <div class="meta" style="margin-bottom:12px;">Created: {{ note.created_at }} &nbsp;|&nbsp; Last edited: {{ note.updated_at }}</div>
    <p class="content" style="font-size:1.05em; line-height:1.7;">{{ note.content }}</p>
    <div class="actions" style="margin-top:20px;">
        <a href="{{ url_for('index') }}" class="btn btn-secondary">← Back</a>
        <a href="{{ url_for('edit_note', note_id=note.id) }}" class="btn btn-primary">Edit</a>
        <form method="POST" action="{{ url_for('delete_note', note_id=note.id) }}" style="margin:0;padding:0;box-shadow:none;" onsubmit="return confirm('Delete this note?')">
            <button type="submit" class="btn btn-danger">Delete</button>
        </form>
    </div>
</div>
{% endblock %}
""")

FORM_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<h2 style="margin-bottom:16px;">{{ form_title }}</h2>
<form method="POST">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" value="{{ note_title }}" placeholder="Enter a title..." required maxlength="200">
    <label for="content">Content</label>
    <textarea id="content" name="content" placeholder="Write your note here...">{{ note_content }}</textarea>
    <div style="display:flex; gap:10px;">
        <button type="submit" class="btn btn-primary">Save Note</button>
        <a href="{{ url_for('index') }}" class="btn btn-secondary">Cancel</a>
    </div>
</form>
{% endblock %}
""")

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def generate_id(notes):
    if not notes:
        return 1
    return max(n["id"] for n in notes) + 1

@app.route("/")
def index():
    notes = load_notes()
    notes_sorted = sorted(notes, key=lambda n: n["updated_at"], reverse=True)
    return render_template_string(INDEX_TEMPLATE, notes=notes_sorted)

@app.route("/note/new", methods=["GET", "POST"])
def new_note():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if not title:
            title = "Untitled"
        notes = load_notes()
        note = {
            "id": generate_id(notes),
            "title": title,
            "content": content,
            "created_at": get_timestamp(),
            "updated_at": get_timestamp(),
        }
        notes.append(note)
        save_notes(notes)
        return redirect(url_for("view_note", note_id=note["id"]))
    return render_template_string(
        FORM_TEMPLATE,
        form_title="New Note",
        note_title="",
        note_content=""
    )

@app.route("/note/<int:note_id>")
def view_note(note_id):
    notes = load_notes()
    note = next((n for n in notes if n["id"] == note_id), None)
    if note is None:
        return "Note not found", 404
    return render_template_string(VIEW_TEMPLATE, note=note)

@app.route("/note/<int:note_id>/edit", methods=["GET", "POST"])
def edit_note(note_id):
    notes = load_notes()
    note = next((n for n in notes if n["id"] == note_id), None)
    if note is None:
        return "Note not found", 404
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if not title:
            title = "Untitled"
        note["title"] = title
        note["content"] = content
        note["updated_at"] = get_timestamp()
        save_notes(notes)
        return redirect(url_for("view_note", note_id=note_id))
    return render_template_string(
        FORM_TEMPLATE,
        form_title="Edit Note",
        note_title=note["title"],
        note_content=note["content"]
    )

@app.route("/note/<int:note_id>/delete", methods=["POST"])
def delete_note(note_id):
    notes = load_notes()
    notes = [n for n in notes if n["id"] != note_id]
    save_notes(notes)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)