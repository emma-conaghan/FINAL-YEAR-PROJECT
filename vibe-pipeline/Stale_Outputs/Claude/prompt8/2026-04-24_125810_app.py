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

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personal Notes</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; }
        nav { background: #4a90d9; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
        nav h1 { color: white; font-size: 1.4rem; }
        nav a { color: white; text-decoration: none; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 4px; font-size: 0.9rem; }
        nav a:hover { background: rgba(255,255,255,0.35); }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 24px; margin-bottom: 20px; }
        .note-card { border-left: 4px solid #4a90d9; }
        .note-card h2 { font-size: 1.2rem; margin-bottom: 8px; color: #2c3e50; }
        .note-card p { color: #666; font-size: 0.95rem; white-space: pre-wrap; word-break: break-word; margin-bottom: 12px; }
        .note-meta { font-size: 0.8rem; color: #999; margin-bottom: 10px; }
        .btn { display: inline-block; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-size: 0.9rem; cursor: pointer; border: none; }
        .btn-primary { background: #4a90d9; color: white; }
        .btn-primary:hover { background: #357abd; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-danger:hover { background: #c0392b; }
        .btn-secondary { background: #95a5a6; color: white; }
        .btn-secondary:hover { background: #7f8c8d; }
        .btn-edit { background: #27ae60; color: white; }
        .btn-edit:hover { background: #219a52; }
        .actions { display: flex; gap: 10px; flex-wrap: wrap; }
        label { display: block; margin-bottom: 6px; font-weight: 600; color: #555; }
        input[type=text], textarea { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; font-family: inherit; }
        input[type=text]:focus, textarea:focus { outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }
        textarea { resize: vertical; min-height: 200px; }
        .form-group { margin-bottom: 18px; }
        .empty-state { text-align: center; padding: 60px 20px; color: #999; }
        .empty-state h2 { font-size: 1.5rem; margin-bottom: 10px; }
        h1.page-title { margin-bottom: 20px; color: #2c3e50; font-size: 1.6rem; }
        .flash { padding: 12px 16px; border-radius: 4px; margin-bottom: 20px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .note-preview { max-height: 120px; overflow: hidden; position: relative; }
        .note-preview::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 40px; background: linear-gradient(transparent, white); }
    </style>
</head>
<body>
    <nav>
        <h1>📝 My Notes</h1>
        <a href="/new">+ New Note</a>
    </nav>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

INDEX_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<h1 class="page-title">All Notes ({{ notes|length }})</h1>
{% if notes %}
    {% for note in notes %}
    <div class="card note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="note-meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <div class="note-preview">
            <p>{{ note['content'] }}</p>
        </div>
        <div class="actions" style="margin-top:14px;">
            <a href="/view/{{ note['id'] }}" class="btn btn-primary">View</a>
            <a href="/edit/{{ note['id'] }}" class="btn btn-edit">Edit</a>
            <form method="POST" action="/delete/{{ note['id'] }}" style="display:inline;" onsubmit="return confirm('Delete this note?')">
                <button type="submit" class="btn btn-danger">Delete</button>
            </form>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="card empty-state">
        <h2>No notes yet</h2>
        <p>Create your first note to get started.</p>
        <br>
        <a href="/new" class="btn btn-primary">Create a Note</a>
    </div>
{% endif %}
{% endblock %}
""")

NEW_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<h1 class="page-title">New Note</h1>
<div class="card">
    <form method="POST" action="/new">
        <div class="form-group">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" placeholder="Enter note title..." required maxlength="200">
        </div>
        <div class="form-group">
            <label for="content">Content</label>
            <textarea id="content" name="content" placeholder="Write your note here..." required></textarea>
        </div>
        <div class="actions">
            <button type="submit" class="btn btn-primary">Save Note</button>
            <a href="/" class="btn btn-secondary">Cancel</a>
        </div>
    </form>
</div>
{% endblock %}
""")

EDIT_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<h1 class="page-title">Edit Note</h1>
<div class="card">
    <form method="POST" action="/edit/{{ note['id'] }}">
        <div class="form-group">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" value="{{ note['title'] }}" required maxlength="200">
        </div>
        <div class="form-group">
            <label for="content">Content</label>
            <textarea id="content" name="content" required>{{ note['content'] }}</textarea>
        </div>
        <div class="actions">
            <button type="submit" class="btn btn-primary">Update Note</button>
            <a href="/view/{{ note['id'] }}" class="btn btn-secondary">Cancel</a>
        </div>
    </form>
</div>
{% endblock %}
""")

VIEW_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="card" style="margin-bottom:16px;">
    <div class="actions">
        <a href="/" class="btn btn-secondary">← Back</a>
        <a href="/edit/{{ note['id'] }}" class="btn btn-edit">Edit</a>
        <form method="POST" action="/delete/{{ note['id'] }}" style="display:inline;" onsubmit="return confirm('Delete this note?')">
            <button type="submit" class="btn btn-danger">Delete</button>
        </form>
    </div>
</div>
<div class="card note-card">
    <h2 style="font-size:1.5rem; margin-bottom:10px;">{{ note['title'] }}</h2>
    <div class="note-meta" style="margin-bottom:16px;">
        Created: {{ note['created_at'] }} | Last updated: {{ note['updated_at'] }}
    </div>
    <p style="line-height:1.7; color:#444; white-space:pre-wrap; word-break:break-word;">{{ note['content'] }}</p>
</div>
{% endblock %}
""")

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
    return render_template_string(NEW_TEMPLATE)

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
            return redirect(url_for("view_note", note_id=note_id))
    conn.close()
    return render_template_string(EDIT_TEMPLATE, note=note)

@app.route("/delete/<int:note_id>", methods=["POST"])
def delete_note(note_id):
    conn = get_db()
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)