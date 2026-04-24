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
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

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
        header {
            background: #4a90d9;
            color: white;
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        header h1 { font-size: 1.5rem; }
        header a { color: white; text-decoration: none; background: #357abd; padding: 8px 16px; border-radius: 4px; }
        header a:hover { background: #2a6099; }
        .container { max-width: 900px; margin: 32px auto; padding: 0 16px; }
        .note-card {
            background: white;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }
        .note-card h2 { font-size: 1.2rem; margin-bottom: 8px; color: #4a90d9; }
        .note-card p { font-size: 0.9rem; color: #666; margin-bottom: 12px; white-space: pre-wrap; word-break: break-word; }
        .note-card .meta { font-size: 0.75rem; color: #999; margin-bottom: 10px; }
        .note-card .actions a {
            display: inline-block;
            margin-right: 8px;
            padding: 6px 14px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.85rem;
        }
        .btn-edit { background: #4a90d9; color: white; }
        .btn-edit:hover { background: #357abd; }
        .btn-delete { background: #e74c3c; color: white; }
        .btn-delete:hover { background: #c0392b; }
        .btn-view { background: #27ae60; color: white; }
        .btn-view:hover { background: #1e8449; }
        form { background: white; padding: 24px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 6px; font-weight: bold; font-size: 0.9rem; }
        form input[type="text"], form textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 1rem;
            margin-bottom: 16px;
            font-family: Arial, sans-serif;
        }
        form textarea { height: 200px; resize: vertical; }
        form input[type="submit"] {
            background: #4a90d9;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
        }
        form input[type="submit"]:hover { background: #357abd; }
        .empty { text-align: center; color: #999; margin-top: 60px; font-size: 1.1rem; }
        .note-view h2 { font-size: 1.6rem; margin-bottom: 8px; color: #4a90d9; }
        .note-view .meta { font-size: 0.8rem; color: #999; margin-bottom: 16px; }
        .note-view .content {
            background: white;
            padding: 20px;
            border-radius: 6px;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.6;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }
        .back-link { display: inline-block; margin-top: 20px; color: #4a90d9; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        .flash { padding: 10px 16px; border-radius: 4px; margin-bottom: 16px; background: #dff0d8; color: #3c763d; border: 1px solid #d6e9c6; }
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
{% block content %}
{% if notes %}
    {% for note_id, note in notes.items() %}
    <div class="note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="meta">Created: {{ note.get('created', 'N/A') }} &nbsp;|&nbsp; Updated: {{ note.get('updated', 'N/A') }}</div>
        <p>{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</p>
        <div class="actions">
            <a class="btn-view" href="{{ url_for('view_note', note_id=note_id) }}">View</a>
            <a class="btn-edit" href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a>
            <a class="btn-delete" href="{{ url_for('delete_note', note_id=note_id) }}" onclick="return confirm('Delete this note?')">Delete</a>
        </div>
    </div>
    {% endfor %}
{% else %}
    <p class="empty">No notes yet. Click <strong>+ New Note</strong> to get started!</p>
{% endif %}
{% endblock %}
""")

FORM_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<h2 style="margin-bottom:20px;">{{ form_title }}</h2>
<form method="POST">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" value="{{ note_title }}" placeholder="Note title" required>
    <label for="content">Content</label>
    <textarea id="content" name="content" placeholder="Write your note here...">{{ note_content }}</textarea>
    <input type="submit" value="{{ submit_label }}">
    <a href="{{ url_for('index') }}" style="margin-left:16px; color:#666; text-decoration:none;">Cancel</a>
</form>
{% endblock %}
""")

VIEW_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="note-view">
    <h2>{{ note['title'] }}</h2>
    <div class="meta">Created: {{ note.get('created', 'N/A') }} &nbsp;|&nbsp; Updated: {{ note.get('updated', 'N/A') }}</div>
    <div class="content">{{ note['content'] }}</div>
    <div style="margin-top:16px;">
        <a class="btn-edit" href="{{ url_for('edit_note', note_id=note_id) }}" style="display:inline-block; padding:8px 16px; background:#4a90d9; color:white; border-radius:4px; text-decoration:none;">Edit</a>
        <a class="btn-delete" href="{{ url_for('delete_note', note_id=note_id) }}" onclick="return confirm('Delete this note?')" style="display:inline-block; margin-left:8px; padding:8px 16px; background:#e74c3c; color:white; border-radius:4px; text-decoration:none;">Delete</a>
    </div>
    <a class="back-link" href="{{ url_for('index') }}">← Back to all notes</a>
</div>
{% endblock %}
""")

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

@app.route("/")
def index():
    notes = load_notes()
    sorted_notes = dict(sorted(notes.items(), key=lambda x: x[1].get("updated", ""), reverse=True))
    return render_template_string(INDEX_TEMPLATE, notes=sorted_notes)

@app.route("/new", methods=["GET", "POST"])
def new_note():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if not title:
            title = "Untitled"
        note_id = str(uuid.uuid4())
        notes = load_notes()
        notes[note_id] = {
            "title": title,
            "content": content,
            "created": now_str(),
            "updated": now_str(),
        }
        save_notes(notes)
        return redirect(url_for("index"))
    return render_template_string(
        FORM_TEMPLATE,
        form_title="New Note",
        note_title="",
        note_content="",
        submit_label="Save Note"
    )

@app.route("/edit/<note_id>", methods=["GET", "POST"])
def edit_note(note_id):
    notes = load_notes()
    if note_id not in notes:
        return redirect(url_for("index"))
    note = notes[note_id]
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if not title:
            title = "Untitled"
        notes[note_id]["title"] = title
        notes[note_id]["content"] = content
        notes[note_id]["updated"] = now_str()
        save_notes(notes)
        return redirect(url_for("index"))
    return render_template_string(
        FORM_TEMPLATE,
        form_title="Edit Note",
        note_title=note["title"],
        note_content=note["content"],
        submit_label="Update Note"
    )

@app.route("/view/<note_id>")
def view_note(note_id):
    notes = load_notes()
    if note_id not in notes:
        return redirect(url_for("index"))
    note = notes[note_id]
    return render_template_string(VIEW_TEMPLATE, note=note, note_id=note_id)

@app.route("/delete/<note_id>")
def delete_note(note_id):
    notes = load_notes()
    if note_id in notes:
        del notes[note_id]
        save_notes(notes)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)