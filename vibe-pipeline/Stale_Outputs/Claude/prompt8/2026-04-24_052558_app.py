from flask import Flask, request, redirect, url_for, render_template_string
import sqlite3
import os

app = Flask(__name__)
DB_PATH = "notes.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

BASE_STYLE = """
<style>
  body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
  h1 { color: #333; }
  a { color: #007bff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .note-card { background: white; border-radius: 6px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .note-card h3 { margin: 0 0 8px 0; }
  .note-card p { margin: 0 0 10px 0; color: #555; white-space: pre-wrap; }
  .note-meta { font-size: 0.8em; color: #999; }
  .actions a { margin-right: 10px; }
  input[type=text], textarea { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; }
  textarea { height: 200px; resize: vertical; }
  input[type=submit] { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 1em; }
  input[type=submit]:hover { background: #0056b3; }
  .btn-danger { color: red; }
  label { display: block; margin: 10px 0 4px; font-weight: bold; }
  nav { margin-bottom: 20px; }
</style>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>My Notes</title>""" + BASE_STYLE + """</head>
<body>
  <h1>📝 My Notes</h1>
  <nav><a href="/new">+ New Note</a></nav>
  {% if notes %}
    {% for note in notes %}
    <div class="note-card">
      <h3><a href="/view/{{ note['id'] }}">{{ note['title'] }}</a></h3>
      <p>{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</p>
      <div class="note-meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
      <div class="actions">
        <a href="/edit/{{ note['id'] }}">Edit</a>
        <a href="/delete/{{ note['id'] }}" class="btn-danger" onclick="return confirm('Delete this note?')">Delete</a>
      </div>
    </div>
    {% endfor %}
  {% else %}
    <p>No notes yet. <a href="/new">Create one!</a></p>
  {% endif %}
</body>
</html>
"""

VIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>{{ note['title'] }}</title>""" + BASE_STYLE + """</head>
<body>
  <h1>{{ note['title'] }}</h1>
  <nav><a href="/">← Back</a> | <a href="/edit/{{ note['id'] }}">Edit</a> | <a href="/delete/{{ note['id'] }}" class="btn-danger" onclick="return confirm('Delete?')">Delete</a></nav>
  <div class="note-card">
    <p>{{ note['content'] }}</p>
    <div class="note-meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
  </div>
</body>
</html>
"""

FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>{{ 'Edit Note' if note else 'New Note' }}</title>""" + BASE_STYLE + """</head>
<body>
  <h1>{{ 'Edit Note' if note else 'New Note' }}</h1>
  <nav><a href="/">← Back</a></nav>
  <form method="POST">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" value="{{ note['title'] if note else '' }}" required>
    <label for="content">Content</label>
    <textarea id="content" name="content" required>{{ note['content'] if note else '' }}</textarea>
    <br><br>
    <input type="submit" value="{{ 'Update Note' if note else 'Save Note' }}">
  </form>
</body>
</html>
"""

@app.route("/")
def index():
    conn = get_db()
    notes = conn.execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()
    conn.close()
    return render_template_string(INDEX_TEMPLATE, notes=notes)

@app.route("/new", methods=["GET", "POST"])
def new_note():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title and content:
            conn = get_db()
            conn.execute("INSERT INTO notes (title, content) VALUES (?, ?)", (title, content))
            conn.commit()
            conn.close()
            return redirect(url_for("index"))
    return render_template_string(FORM_TEMPLATE, note=None)

@app.route("/view/<int:note_id>")
def view_note(note_id):
    conn = get_db()
    note = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    conn.close()
    if note is None:
        return "Note not found", 404
    return render_template_string(VIEW_TEMPLATE, note=note)

@app.route("/edit/<int:note_id>", methods=["GET", "POST"])
def edit_note(note_id):
    conn = get_db()
    note = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if note is None:
        conn.close()
        return "Note not found", 404
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title and content:
            conn.execute(
                "UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, content, note_id)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("index"))
    conn.close()
    return render_template_string(FORM_TEMPLATE, note=note)

@app.route("/delete/<int:note_id>")
def delete_note(note_id):
    conn = get_db()
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)