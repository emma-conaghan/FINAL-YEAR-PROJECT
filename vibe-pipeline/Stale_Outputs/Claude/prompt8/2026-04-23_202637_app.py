from flask import Flask, request, redirect, url_for, render_template_string
import os
import json
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

BASE_STYLE = """
<style>
  body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; color: #333; }
  h1 { color: #2c3e50; }
  a { color: #3498db; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .note-card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .note-card h2 { margin: 0 0 10px 0; color: #2c3e50; }
  .note-card p { margin: 0 0 10px 0; white-space: pre-wrap; }
  .note-card small { color: #999; }
  .actions { margin-top: 10px; }
  .actions a { margin-right: 10px; }
  .btn { display: inline-block; padding: 8px 16px; background: #3498db; color: white; border-radius: 4px; border: none; cursor: pointer; font-size: 14px; }
  .btn:hover { background: #2980b9; color: white; }
  .btn-danger { background: #e74c3c; }
  .btn-danger:hover { background: #c0392b; }
  .btn-success { background: #2ecc71; }
  .btn-success:hover { background: #27ae60; }
  input[type=text], textarea { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; font-family: Arial, sans-serif; }
  textarea { height: 200px; resize: vertical; }
  label { display: block; margin: 10px 0 5px 0; font-weight: bold; }
  .form-group { margin-bottom: 15px; }
  nav { margin-bottom: 30px; }
  .empty-msg { color: #999; font-style: italic; }
</style>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>My Notes</title>""" + BASE_STYLE + """</head>
<body>
  <nav><a href="/">My Notes</a> | <a href="/new" class="btn btn-success">+ New Note</a></nav>
  <h1>My Notes</h1>
  {% if notes %}
    {% for note in notes|reverse %}
    <div class="note-card">
      <h2>{{ note.title }}</h2>
      <p>{{ note.content }}</p>
      <small>Created: {{ note.created }}</small>
      <div class="actions">
        <a href="/edit/{{ note.id }}" class="btn">Edit</a>
        <form method="post" action="/delete/{{ note.id }}" style="display:inline;">
          <button type="submit" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</button>
        </form>
      </div>
    </div>
    {% endfor %}
  {% else %}
    <p class="empty-msg">No notes yet. <a href="/new">Create your first note!</a></p>
  {% endif %}
</body>
</html>
"""

NEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>New Note</title>""" + BASE_STYLE + """</head>
<body>
  <nav><a href="/">My Notes</a></nav>
  <h1>New Note</h1>
  <form method="post" action="/new">
    <div class="form-group">
      <label for="title">Title</label>
      <input type="text" id="title" name="title" placeholder="Enter note title..." required>
    </div>
    <div class="form-group">
      <label for="content">Content</label>
      <textarea id="content" name="content" placeholder="Write your note here..."></textarea>
    </div>
    <button type="submit" class="btn btn-success">Save Note</button>
    <a href="/" class="btn">Cancel</a>
  </form>
</body>
</html>
"""

EDIT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Edit Note</title>""" + BASE_STYLE + """</head>
<body>
  <nav><a href="/">My Notes</a></nav>
  <h1>Edit Note</h1>
  <form method="post" action="/edit/{{ note.id }}">
    <div class="form-group">
      <label for="title">Title</label>
      <input type="text" id="title" name="title" value="{{ note.title }}" required>
    </div>
    <div class="form-group">
      <label for="content">Content</label>
      <textarea id="content" name="content">{{ note.content }}</textarea>
    </div>
    <button type="submit" class="btn btn-success">Update Note</button>
    <a href="/" class="btn">Cancel</a>
  </form>
</body>
</html>
"""

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
            note_id = str(int(time.time() * 1000))
            created = time.strftime("%Y-%m-%d %H:%M:%S")
            notes.append({
                "id": note_id,
                "title": title,
                "content": content,
                "created": created
            })
            save_notes(notes)
        return redirect(url_for("index"))
    return render_template_string(NEW_TEMPLATE)

@app.route("/edit/<note_id>", methods=["GET", "POST"])
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
    return render_template_string(EDIT_TEMPLATE, note=note)

@app.route("/delete/<note_id>", methods=["POST"])
def delete_note(note_id):
    notes = load_notes()
    notes = [n for n in notes if n["id"] != note_id]
    save_notes(notes)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)