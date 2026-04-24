from flask import Flask, render_template_string, request, redirect, url_for
import sqlite3
import os

app = Flask(__name__)
DATABASE = 'notes.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ page_title | default("Personal Notes") }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        header {
            background: #4a90d9;
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        header .container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        header h1 { font-size: 1.5em; }
        header a { color: white; text-decoration: none; font-weight: bold; }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            background: #4a90d9;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 1em;
        }
        .btn:hover { background: #357abd; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-secondary { background: #95a5a6; }
        .btn-secondary:hover { background: #7f8c8d; }
        .note-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        .note-card:hover { box-shadow: 0 3px 8px rgba(0,0,0,0.15); }
        .note-card h2 { margin-bottom: 8px; color: #2c3e50; }
        .note-card .preview {
            color: #666;
            margin-bottom: 10px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .note-card .meta { font-size: 0.85em; color: #999; }
        .note-card .actions { margin-top: 10px; }
        .note-card .actions a { margin-right: 10px; font-size: 0.9em; }
        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        .form-group input[type="text"],
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            font-family: inherit;
        }
        .form-group textarea { min-height: 300px; resize: vertical; }
        .form-group input[type="text"]:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #4a90d9;
            box-shadow: 0 0 0 2px rgba(74,144,217,0.2);
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state h2 { margin-bottom: 10px; }
        .note-content {
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .note-content h1 { margin-bottom: 10px; color: #2c3e50; }
        .note-content .meta { color: #999; font-size: 0.9em; margin-bottom: 20px; }
        .note-content .body { white-space: pre-wrap; line-height: 1.8; }
        .note-content .actions { margin-top: 25px; padding-top: 15px; border-top: 1px solid #eee; }
        .note-content .actions a { margin-right: 10px; }
        .flash {
            padding: 12px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="/">📝 Personal Notes</a></h1>
            <a href="/notes/new" class="btn">+ New Note</a>
        </div>
    </header>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

INDEX_TEMPLATE = '''
{% extends base %}
{% block content %}
{% if notes %}
    {% for note in notes %}
    <div class="note-card">
        <h2><a href="/notes/{{ note.id }}" style="text-decoration:none;color:inherit;">{{ note.title }}</a></h2>
        <div class="preview">{{ note.content[:150] }}</div>
        <div class="meta">Created: {{ note.created_at }} | Updated: {{ note.updated_at }}</div>
        <div class="actions">
            <a href="/notes/{{ note.id }}" class="btn" style="padding:5px 12px;font-size:0.85em;">View</a>
            <a href="/notes/{{ note.id }}/edit" class="btn btn-secondary" style="padding:5px 12px;font-size:0.85em;">Edit</a>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="empty-state">
        <h2>No notes yet</h2>
        <p>Create your first note to get started!</p>
        <br>
        <a href="/notes/new" class="btn">+ Create Note</a>
    </div>
{% endif %}
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends base %}
{% block content %}
<div class="note-content">
    <h1>{{ note.title }}</h1>
    <div class="meta">Created: {{ note.created_at }} | Updated: {{ note.updated_at }}</div>
    <div class="body">{{ note.content }}</div>
    <div class="actions">
        <a href="/notes/{{ note.id }}/edit" class="btn">Edit</a>
        <a href="/" class="btn btn-secondary">Back to Notes</a>
        <form action="/notes/{{ note.id }}/delete" method="POST" style="display:inline;" onsubmit="return confirm('Are you sure you want to delete this note?');">
            <button type="submit" class="btn btn-danger">Delete</button>
        </form>
    </div>
</div>
{% endblock %}
'''

FORM_TEMPLATE = '''
{% extends base %}
{% block content %}
<div class="note-content">
    <h1>{{ form_title }}</h1>
    <form method="POST">
        <div class="form-group">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" value="{{ note_title }}" required placeholder="Enter note title...">
        </div>
        <div class="form-group">
            <label for="content">Content</label>
            <textarea id="content" name="content" required placeholder="Write your note here...">{{ note_content }}</textarea>
        </div>
        <button type="submit" class="btn">{{ submit_text }}</button>
        <a href="/" class="btn btn-secondary" style="margin-left:10px;">Cancel</a>
    </form>
</div>
{% endblock %}
'''

@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render_template_string(INDEX_TEMPLATE, base=BASE_TEMPLATE, notes=notes, page_title="Personal Notes")

@app.route('/notes/new', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            conn = get_db()
            cursor = conn.execute('INSERT INTO notes (title, content) VALUES (?, ?)', (title, content))
            conn.commit()
            note_id = cursor.lastrowid
            conn.close()
            return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(
        FORM_TEMPLATE,
        base=BASE_TEMPLATE,
        form_title="Create New Note",
        note_title="",
        note_content="",
        submit_text="Save Note",
        page_title="New Note"
    )

@app.route('/notes/<int:note_id>')
def view_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect('/')
    return render_template_string(VIEW_TEMPLATE, base=BASE_TEMPLATE, note=note, page_title=note['title'])

@app.route('/notes/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    if note is None:
        conn.close()
        return redirect('/')
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            conn.execute(
                'UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (title, content, note_id)
            )
            conn.commit()
            conn.close()
            return redirect(url_for('view_note', note_id=note_id))
    conn.close()
    return render_template_string(
        FORM_TEMPLATE,
        base=BASE_TEMPLATE,
        form_title="Edit Note",
        note_title=note['title'],
        note_content=note['content'],
        submit_text="Update Note",
        page_title="Edit: " + note['title']
    )

@app.route('/notes/<int:note_id>/delete', methods=['POST'])
def delete_note(note_id):
    conn = get_db()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    return redirect('/')

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)