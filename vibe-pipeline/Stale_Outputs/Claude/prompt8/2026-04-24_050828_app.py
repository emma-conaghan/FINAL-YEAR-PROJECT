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
    with get_db() as conn:
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
        .navbar { background: #4a90d9; padding: 1rem 2rem; display: flex; align-items: center; justify-content: space-between; }
        .navbar h1 { color: white; font-size: 1.5rem; }
        .navbar a { color: white; text-decoration: none; background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 4px; font-size: 0.9rem; }
        .navbar a:hover { background: rgba(255,255,255,0.35); }
        .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
        .note-card { border-left: 4px solid #4a90d9; }
        .note-card h2 { font-size: 1.2rem; margin-bottom: 0.5rem; color: #2c3e50; }
        .note-card p { color: #555; white-space: pre-wrap; word-break: break-word; margin-bottom: 0.8rem; }
        .note-meta { font-size: 0.78rem; color: #999; margin-bottom: 0.8rem; }
        .actions { display: flex; gap: 0.5rem; }
        .btn { display: inline-block; padding: 0.4rem 1rem; border-radius: 4px; text-decoration: none; font-size: 0.85rem; border: none; cursor: pointer; }
        .btn-primary { background: #4a90d9; color: white; }
        .btn-primary:hover { background: #357abd; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-danger:hover { background: #c0392b; }
        .btn-secondary { background: #95a5a6; color: white; }
        .btn-secondary:hover { background: #7f8c8d; }
        form label { display: block; font-weight: 600; margin-bottom: 0.3rem; color: #555; }
        form input[type=text], form textarea { width: 100%; padding: 0.6rem; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; margin-bottom: 1rem; font-family: inherit; }
        form textarea { min-height: 200px; resize: vertical; }
        form input[type=text]:focus, form textarea:focus { outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }
        .empty-state { text-align: center; padding: 3rem; color: #aaa; }
        .empty-state p { margin-bottom: 1rem; font-size: 1.1rem; }
        .flash { padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 1rem; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .note-preview { max-height: 100px; overflow: hidden; position: relative; }
        .note-preview::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 30px; background: linear-gradient(transparent, white); }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>📝 Personal Notes</h1>
        <a href="{{ url_for('new_note') }}">+ New Note</a>
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
    <p style="color:#777; margin-bottom:1rem;">{{ notes|length }} note{{ 's' if notes|length != 1 else '' }} saved</p>
    {% for note in notes %}
    <div class="card note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="note-meta">Created: {{ note['created_at'] }}{% if note['updated_at'] != note['created_at'] %} &bull; Updated: {{ note['updated_at'] }}{% endif %}</div>
        <div class="note-preview">
            <p>{{ note['content'] }}</p>
        </div>
        <div class="actions" style="margin-top:0.8rem;">
            <a href="{{ url_for('view_note', note_id=note['id']) }}" class="btn btn-primary">View</a>
            <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn btn-secondary">Edit</a>
            <form method="POST" action="{{ url_for('delete_note', note_id=note['id']) }}" style="display:inline;" onsubmit="return confirm('Delete this note?')">
                <button type="submit" class="btn btn-danger">Delete</button>
            </form>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="card empty-state">
        <p>No notes yet. Create your first note!</p>
        <a href="{{ url_for('new_note') }}" class="btn btn-primary">Create Note</a>
    </div>
{% endif %}
{% endblock %}""")

FORM_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="card">
    <h2 style="margin-bottom:1.2rem; color:#2c3e50;">{{ 'Edit Note' if note else 'New Note' }}</h2>
    <form method="POST">
        <label for="title">Title</label>
        <input type="text" id="title" name="title" placeholder="Enter note title..." value="{{ note['title'] if note else '' }}" required>
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here...">{{ note['content'] if note else '' }}</textarea>
        <div style="display:flex; gap:0.5rem;">
            <button type="submit" class="btn btn-primary">{{ 'Save Changes' if note else 'Save Note' }}</button>
            <a href="{{ url_for('index') }}" class="btn btn-secondary">Cancel</a>
        </div>
    </form>
</div>
{% endblock %}""")

VIEW_TEMPLATE = BASE_TEMPLATE.replace("{% block content %}{% endblock %}", """
{% block content %}
<div class="card note-card">
    <h2 style="font-size:1.5rem; margin-bottom:0.5rem;">{{ note['title'] }}</h2>
    <div class="note-meta">Created: {{ note['created_at'] }}{% if note['updated_at'] != note['created_at'] %} &bull; Updated: {{ note['updated_at'] }}{% endif %}</div>
    <hr style="border:none; border-top:1px solid #eee; margin:1rem 0;">
    <p style="white-space:pre-wrap; line-height:1.7; color:#444;">{{ note['content'] }}</p>
    <div class="actions" style="margin-top:1.5rem;">
        <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn btn-primary">Edit</a>
        <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Notes</a>
        <form method="POST" action="{{ url_for('delete_note', note_id=note['id']) }}" style="display:inline;" onsubmit="return confirm('Delete this note?')">
            <button type="submit" class="btn btn-danger">Delete</button>
        </form>
    </div>
</div>
{% endblock %}""")

@app.route("/")
def index():
    db = get_db()
    notes = db.execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()
    db.close()
    return render_template_string(INDEX_TEMPLATE, notes=notes)

@app.route("/note/new", methods=["GET", "POST"])
def new_note():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title:
            db = get_db()
            db.execute("INSERT INTO notes (title, content) VALUES (?, ?)", (title, content))
            db.commit()
            db.close()
            return redirect(url_for("index"))
    return render_template_string(FORM_TEMPLATE, note=None)

@app.route("/note/<int:note_id>")
def view_note(note_id):
    db = get_db()
    note = db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    db.close()
    if note is None:
        return redirect(url_for("index"))
    return render_template_string(VIEW_TEMPLATE, note=note)

@app.route("/note/<int:note_id>/edit", methods=["GET", "POST"])
def edit_note(note_id):
    db = get_db()
    note = db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if note is None:
        db.close()
        return redirect(url_for("index"))
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        if title:
            db.execute(
                "UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, content, note_id)
            )
            db.commit()
            db.close()
            return redirect(url_for("view_note", note_id=note_id))
    db.close()
    return render_template_string(FORM_TEMPLATE, note=note)

@app.route("/note/<int:note_id>/delete", methods=["POST"])
def delete_note(note_id):
    db = get_db()
    db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    db.commit()
    db.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)