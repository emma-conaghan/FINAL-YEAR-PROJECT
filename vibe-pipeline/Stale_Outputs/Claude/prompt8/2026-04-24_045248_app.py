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
    h1, h2 { color: #333; }
    a { color: #007bff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .note-card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .note-card h3 { margin: 0 0 10px 0; color: #333; }
    .note-card p { color: #666; white-space: pre-wrap; }
    .note-meta { font-size: 0.8em; color: #999; margin-top: 10px; }
    .btn { display: inline-block; padding: 8px 16px; border-radius: 4px; border: none; cursor: pointer; font-size: 14px; }
    .btn-primary { background: #007bff; color: white; }
    .btn-danger { background: #dc3545; color: white; }
    .btn-secondary { background: #6c757d; color: white; }
    .btn:hover { opacity: 0.85; }
    form input[type=text], form textarea {
        width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box;
        border: 1px solid #ccc; border-radius: 4px; font-size: 14px;
    }
    form textarea { height: 200px; resize: vertical; font-family: Arial, sans-serif; }
    .actions { margin-top: 10px; }
    .actions a { margin-right: 10px; }
    nav { margin-bottom: 20px; }
    .empty-state { text-align: center; color: #999; padding: 40px; }
</style>
"""

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>My Notes</title>""" + BASE_STYLE + """</head>
<body>
    <nav><a href="/">&#128221; My Notes</a></nav>
    <h1>My Notes</h1>
    <a href="/notes/new" class="btn btn-primary">+ New Note</a>
    <div style="margin-top: 20px;">
        {% if notes %}
            {% for note in notes %}
            <div class="note-card">
                <h3><a href="/notes/{{ note['id'] }}">{{ note['title'] }}</a></h3>
                <p>{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</p>
                <div class="note-meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
                <div class="actions">
                    <a href="/notes/{{ note['id'] }}/edit" class="btn btn-secondary">Edit</a>
                    <form method="POST" action="/notes/{{ note['id'] }}/delete" style="display:inline;">
                        <button type="submit" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</button>
                    </form>
                </div>
            </div>
            {% endfor %}
        {% else %}
            <div class="empty-state">
                <p>No notes yet. <a href="/notes/new">Create your first note!</a></p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

VIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>{{ note['title'] }}</title>""" + BASE_STYLE + """</head>
<body>
    <nav><a href="/">&#128221; My Notes</a> &gt; {{ note['title'] }}</nav>
    <div class="note-card">
        <h1>{{ note['title'] }}</h1>
        <p>{{ note['content'] }}</p>
        <div class="note-meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <div class="actions" style="margin-top: 15px;">
            <a href="/notes/{{ note['id'] }}/edit" class="btn btn-secondary">Edit</a>
            <form method="POST" action="/notes/{{ note['id'] }}/delete" style="display:inline;">
                <button type="submit" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</button>
            </form>
            <a href="/" class="btn btn-primary" style="margin-left:10px;">Back to Notes</a>
        </div>
    </div>
</body>
</html>
"""

FORM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>{{ form_title }}</title>""" + BASE_STYLE + """</head>
<body>
    <nav><a href="/">&#128221; My Notes</a> &gt; {{ form_title }}</nav>
    <h1>{{ form_title }}</h1>
    <div class="note-card">
        <form method="POST" action="{{ action }}">
            <label for="title"><strong>Title</strong></label><br>
            <input type="text" id="title" name="title" placeholder="Enter note title..." value="{{ title }}" required>
            <label for="content"><strong>Content</strong></label><br>
            <textarea id="content" name="content" placeholder="Write your note here...">{{ content }}</textarea>
            <div style="margin-top: 10px;">
                <button type="submit" class="btn btn-primary">Save Note</button>
                <a href="/" class="btn btn-secondary" style="margin-left: 10px;">Cancel</a>
            </div>
        </form>
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    conn = get_db()
    notes = conn.execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()
    conn.close()
    return render_template_string(INDEX_TEMPLATE, notes=notes)

@app.route("/notes/new", methods=["GET", "POST"])
def new_note():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title:
            conn = get_db()
            conn.execute("INSERT INTO notes (title, content) VALUES (?, ?)", (title, content))
            conn.commit()
            conn.close()
            return redirect(url_for("index"))
    return render_template_string(FORM_TEMPLATE, form_title="New Note", action="/notes/new", title="", content="")

@app.route("/notes/<int:note_id>")
def view_note(note_id):
    conn = get_db()
    note = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    conn.close()
    if note is None:
        return "Note not found", 404
    return render_template_string(VIEW_TEMPLATE, note=note)

@app.route("/notes/<int:note_id>/edit", methods=["GET", "POST"])
def edit_note(note_id):
    conn = get_db()
    note = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if note is None:
        conn.close()
        return "Note not found", 404
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title:
            conn.execute(
                "UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, content, note_id)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("view_note", note_id=note_id))
    conn.close()
    return render_template_string(
        FORM_TEMPLATE,
        form_title="Edit Note",
        action=f"/notes/{note_id}/edit",
        title=note["title"],
        content=note["content"]
    )

@app.route("/notes/<int:note_id>/delete", methods=["POST"])
def delete_note(note_id):
    conn = get_db()
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)