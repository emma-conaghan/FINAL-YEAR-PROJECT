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
        .navbar { background: #4a6fa5; color: white; padding: 15px 30px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        .navbar h1 { font-size: 1.5rem; }
        .navbar a { color: white; text-decoration: none; background: #3a5a8a; padding: 8px 16px; border-radius: 5px; font-size: 0.9rem; transition: background 0.2s; }
        .navbar a:hover { background: #2d4870; }
        .container { max-width: 900px; margin: 30px auto; padding: 0 20px; }
        .card { background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .note-card { border-left: 4px solid #4a6fa5; transition: transform 0.1s; }
        .note-card:hover { transform: translateX(3px); }
        .note-title { font-size: 1.2rem; font-weight: 600; color: #4a6fa5; margin-bottom: 8px; }
        .note-content { color: #555; font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; max-height: 100px; overflow: hidden; text-overflow: ellipsis; }
        .note-meta { font-size: 0.78rem; color: #999; margin-top: 10px; }
        .note-actions { display: flex; gap: 10px; margin-top: 12px; }
        .btn { padding: 6px 14px; border-radius: 5px; border: none; cursor: pointer; font-size: 0.85rem; text-decoration: none; display: inline-block; transition: opacity 0.2s; }
        .btn:hover { opacity: 0.85; }
        .btn-primary { background: #4a6fa5; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        .form-group { margin-bottom: 18px; }
        .form-group label { display: block; font-weight: 600; margin-bottom: 6px; color: #555; }
        .form-group input[type=text], .form-group textarea { width: 100%; padding: 10px 14px; border: 1px solid #ddd; border-radius: 6px; font-size: 0.95rem; font-family: inherit; transition: border-color 0.2s; }
        .form-group input[type=text]:focus, .form-group textarea:focus { outline: none; border-color: #4a6fa5; box-shadow: 0 0 0 3px rgba(74,111,165,0.15); }
        .form-group textarea { resize: vertical; min-height: 180px; }
        .empty-state { text-align: center; padding: 60px 20px; color: #aaa; }
        .empty-state .icon { font-size: 3rem; margin-bottom: 15px; }
        h2 { margin-bottom: 20px; color: #333; font-size: 1.3rem; }
        .full-content { white-space: pre-wrap; line-height: 1.8; color: #444; }
        .alert { padding: 12px 16px; border-radius: 6px; margin-bottom: 20px; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .notes-grid { display: grid; gap: 15px; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📝 Personal Notes</h1>
        <a href="/new">+ New Note</a>
    </div>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

INDEX_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
{% if notes %}
<h2>All Notes ({{ notes|length }})</h2>
<div class="notes-grid">
{% for note in notes %}
<div class="card note-card">
    <div class="note-title">{{ note['title'] }}</div>
    <div class="note-content">{{ note['content'] }}</div>
    <div class="note-meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
    <div class="note-actions">
        <a href="/view/{{ note['id'] }}" class="btn btn-primary">View</a>
        <a href="/edit/{{ note['id'] }}" class="btn btn-secondary">Edit</a>
        <form method="post" action="/delete/{{ note['id'] }}" style="display:inline;" onsubmit="return confirm('Delete this note?')">
            <button type="submit" class="btn btn-danger">Delete</button>
        </form>
    </div>
</div>
{% endfor %}
</div>
{% else %}
<div class="card">
    <div class="empty-state">
        <div class="icon">📄</div>
        <p>No notes yet. Click <strong>+ New Note</strong> to get started!</p>
    </div>
</div>
{% endif %}
{% endblock %}""")

FORM_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="card">
    <h2>{{ 'Edit Note' if note else 'New Note' }}</h2>
    <form method="post" action="{{ '/edit/' + note['id']|string if note else '/new' }}">
        <div class="form-group">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" value="{{ note['title'] if note else '' }}" placeholder="Enter note title..." required>
        </div>
        <div class="form-group">
            <label for="content">Content</label>
            <textarea id="content" name="content" placeholder="Write your note here...">{{ note['content'] if note else '' }}</textarea>
        </div>
        <div style="display:flex;gap:10px;">
            <button type="submit" class="btn btn-success">{{ 'Update Note' if note else 'Save Note' }}</button>
            <a href="/" class="btn btn-secondary">Cancel</a>
        </div>
    </form>
</div>
{% endblock %}""")

VIEW_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="card">
    <div class="note-title" style="font-size:1.5rem;margin-bottom:12px;">{{ note['title'] }}</div>
    <div class="note-meta" style="margin-bottom:20px;">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
    <div class="full-content">{{ note['content'] }}</div>
    <div class="note-actions" style="margin-top:25px;">
        <a href="/edit/{{ note['id'] }}" class="btn btn-secondary">Edit</a>
        <a href="/" class="btn btn-primary">Back to Notes</a>
        <form method="post" action="/delete/{{ note['id'] }}" style="display:inline;" onsubmit="return confirm('Delete this note?')">
            <button type="submit" class="btn btn-danger">Delete</button>
        </form>
    </div>
</div>
{% endblock %}""")

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
        if title:
            conn = get_db()
            conn.execute("INSERT INTO notes (title, content) VALUES (?, ?)", (title, content))
            conn.commit()
            conn.close()
            return redirect(url_for("index"))
    return render_template_string(FORM_TEMPLATE, note=None)

@app.route("/edit/<int:note_id>", methods=["GET", "POST"])
def edit_note(note_id):
    conn = get_db()
    note = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if note is None:
        conn.close()
        return redirect(url_for("index"))
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
            return redirect(url_for("index"))
    conn.close()
    return render_template_string(FORM_TEMPLATE, note=note)

@app.route("/view/<int:note_id>")
def view_note(note_id):
    conn = get_db()
    note = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for("index"))
    return render_template_string(VIEW_TEMPLATE, note=note)

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