from flask import Flask, request, redirect, url_for, render_template_string
import os
import json
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

BASE_STYLE = """
<style>
  body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h1, h2 { color: #333; }
  a { color: #0066cc; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .note-card { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
  .note-title { font-size: 1.2em; font-weight: bold; color: #222; }
  .note-meta { font-size: 0.85em; color: #888; margin: 4px 0 8px 0; }
  .note-preview { color: #555; white-space: pre-wrap; }
  .btn { display: inline-block; padding: 8px 16px; border-radius: 4px; border: none; cursor: pointer; font-size: 0.9em; }
  .btn-primary { background: #0066cc; color: white; }
  .btn-danger { background: #cc0000; color: white; }
  .btn-secondary { background: #888; color: white; }
  .btn:hover { opacity: 0.85; }
  input[type=text], textarea { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-size: 1em; }
  textarea { height: 200px; resize: vertical; font-family: Arial, sans-serif; }
  .form-group { margin-bottom: 12px; }
  label { display: block; margin-bottom: 4px; font-weight: bold; color: #333; }
  .actions { margin-top: 10px; }
  .actions a, .actions form { display: inline; margin-right: 8px; }
  .empty { color: #888; font-style: italic; }
  nav { margin-bottom: 20px; }
</style>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>My Notes</title>""" + BASE_STYLE + """</head>
<body>
  <h1>📝 My Notes</h1>
  <nav><a href="/new" class="btn btn-primary">+ New Note</a></nav>
  {% if notes %}
    {% for note_id, note in notes.items() | sort(attribute='1.updated_at', reverse=True) %}
    <div class="note-card">
      <div class="note-title">{{ note.title }}</div>
      <div class="note-meta">Last updated: {{ note.updated_at }}</div>
      <div class="note-preview">{{ note.content[:200] }}{% if note.content | length > 200 %}...{% endif %}</div>
      <div class="actions">
        <a href="/view/{{ note_id }}" class="btn btn-secondary">View</a>
        <a href="/edit/{{ note_id }}" class="btn btn-primary">Edit</a>
        <form method="POST" action="/delete/{{ note_id }}" onsubmit="return confirm('Delete this note?');">
          <button type="submit" class="btn btn-danger">Delete</button>
        </form>
      </div>
    </div>
    {% endfor %}
  {% else %}
    <p class="empty">No notes yet. Create your first note!</p>
  {% endif %}
</body>
</html>
"""

FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>{{ page_title }}</title>""" + BASE_STYLE + """</head>
<body>
  <h1>{{ page_title }}</h1>
  <form method="POST" action="{{ action }}">
    <div class="form-group">
      <label for="title">Title</label>
      <input type="text" id="title" name="title" value="{{ note.title if note else '' }}" required placeholder="Enter note title..." />
    </div>
    <div class="form-group">
      <label for="content">Content</label>
      <textarea id="content" name="content" placeholder="Write your note here...">{{ note.content if note else '' }}</textarea>
    </div>
    <button type="submit" class="btn btn-primary">Save Note</button>
    <a href="/" class="btn btn-secondary" style="margin-left:8px;">Cancel</a>
  </form>
</body>
</html>
"""

VIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>{{ note.title }}</title>""" + BASE_STYLE + """</head>
<body>
  <h1>{{ note.title }}</h1>
  <p style="color:#888; font-size:0.9em;">Created: {{ note.created_at }} &nbsp;|&nbsp; Updated: {{ note.updated_at }}</p>
  <div class="note-card">
    <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; color: #333;">{{ note.content }}</pre>
  </div>
  <a href="/edit/{{ note_id }}" class="btn btn-primary">Edit</a>
  <a href="/" class="btn btn-secondary" style="margin-left:8px;">Back to Notes</a>
</body>
</html>
"""

@app.route("/")
def index():
    notes = load_notes()
    return render_template_string(INDEX_TEMPLATE, notes=notes)

@app.route("/new", methods=["GET"])
def new_note_form():
    return render_template_string(FORM_TEMPLATE, page_title="New Note", action="/new", note=None)

@app.route("/new", methods=["POST"])
def create_note():
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()
    if not title:
        return redirect(url_for("new_note_form"))
    notes = load_notes()
    note_id = str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notes[note_id] = {
        "title": title,
        "content": content,
        "created_at": now,
        "updated_at": now
    }
    save_notes(notes)
    return redirect(url_for("index"))

@app.route("/view/<note_id>")
def view_note(note_id):
    notes = load_notes()
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    return render_template_string(VIEW_TEMPLATE, note=note, note_id=note_id)

@app.route("/edit/<note_id>", methods=["GET"])
def edit_note_form(note_id):
    notes = load_notes()
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    return render_template_string(FORM_TEMPLATE, page_title="Edit Note", action=f"/edit/{note_id}", note=note)

@app.route("/edit/<note_id>", methods=["POST"])
def update_note(note_id):
    title = request.form.get("title", "").strip()
    content = request.form.get("content", "").strip()
    if not title:
        return redirect(url_for("edit_note_form", note_id=note_id))
    notes = load_notes()
    if note_id not in notes:
        return "Note not found", 404
    notes[note_id]["title"] = title
    notes[note_id]["content"] = content
    notes[note_id]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_notes(notes)
    return redirect(url_for("index"))

@app.route("/delete/<note_id>", methods=["POST"])
def delete_note(note_id):
    notes = load_notes()
    if note_id in notes:
        del notes[note_id]
        save_notes(notes)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)