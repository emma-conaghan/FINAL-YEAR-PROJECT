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
    <title>Personal Notes</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
        }
        header .container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        header h1 { font-size: 1.8em; }
        header a {
            color: white;
            text-decoration: none;
            background-color: #27ae60;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        }
        header a:hover { background-color: #2ecc71; }
        .note-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        .note-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
        .note-card h2 {
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .note-card p {
            color: #666;
            white-space: pre-wrap;
            margin-bottom: 10px;
        }
        .note-card .meta {
            font-size: 0.85em;
            color: #999;
            margin-bottom: 10px;
        }
        .note-card .actions a {
            text-decoration: none;
            padding: 5px 15px;
            border-radius: 4px;
            font-size: 0.9em;
            margin-right: 5px;
        }
        .btn-view { background-color: #3498db; color: white; }
        .btn-view:hover { background-color: #2980b9; }
        .btn-edit { background-color: #f39c12; color: white; }
        .btn-edit:hover { background-color: #e67e22; }
        .btn-delete {
            background-color: #e74c3c;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 4px;
            font-size: 0.9em;
            cursor: pointer;
        }
        .btn-delete:hover { background-color: #c0392b; }
        form.inline { display: inline; }
        .form-container {
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }
        .form-group input[type="text"],
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            font-family: inherit;
        }
        .form-group input[type="text"]:focus,
        .form-group textarea:focus {
            border-color: #3498db;
            outline: none;
        }
        .form-group textarea { min-height: 200px; resize: vertical; }
        .btn-submit {
            background-color: #27ae60;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            font-weight: bold;
        }
        .btn-submit:hover { background-color: #2ecc71; }
        .btn-cancel {
            text-decoration: none;
            color: #666;
            margin-left: 15px;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state h2 { margin-bottom: 10px; }
        .view-note {
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .view-note h2 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1.8em;
        }
        .view-note .meta {
            color: #999;
            font-size: 0.9em;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .view-note .content {
            white-space: pre-wrap;
            line-height: 1.8;
        }
        .view-note .actions {
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #3498db;
            text-decoration: none;
        }
        .back-link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>📝 Personal Notes</h1>
            <a href="{{ url_for('create_note') }}">+ New Note</a>
        </div>
    </header>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

INDEX_TEMPLATE = '''
{% extends "base" %}
{% block content %}
{% if notes %}
    {% for note in notes %}
    <div class="note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <p>{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</p>
        <div class="actions">
            <a href="{{ url_for('view_note', note_id=note['id']) }}" class="btn-view">View</a>
            <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
            <form class="inline" method="POST" action="{{ url_for('delete_note', note_id=note['id']) }}" onsubmit="return confirm('Are you sure you want to delete this note?');">
                <button type="submit" class="btn-delete">Delete</button>
            </form>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="empty-state">
        <h2>No notes yet</h2>
        <p>Click "+ New Note" to create your first note!</p>
    </div>
{% endif %}
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
<div class="form-container">
    <h2 style="margin-bottom: 20px; color: #2c3e50;">Create New Note</h2>
    <form method="POST">
        <div class="form-group">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" placeholder="Enter note title..." required>
        </div>
        <div class="form-group">
            <label for="content">Content</label>
            <textarea id="content" name="content" placeholder="Write your note here..." required></textarea>
        </div>
        <button type="submit" class="btn-submit">Save Note</button>
        <a href="{{ url_for('index') }}" class="btn-cancel">Cancel</a>
    </form>
</div>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<a href="{{ url_for('view_note', note_id=note['id']) }}" class="back-link">← Back to Note</a>
<div class="form-container">
    <h2 style="margin-bottom: 20px; color: #2c3e50;">Edit Note</h2>
    <form method="POST">
        <div class="form-group">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" value="{{ note['title'] }}" required>
        </div>
        <div class="form-group">
            <label for="content">Content</label>
            <textarea id="content" name="content" required>{{ note['content'] }}</textarea>
        </div>
        <button type="submit" class="btn-submit">Update Note</button>
        <a href="{{ url_for('view_note', note_id=note['id']) }}" class="btn-cancel">Cancel</a>
    </form>
</div>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
<div class="view-note">
    <h2>{{ note['title'] }}</h2>
    <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
    <div class="content">{{ note['content'] }}</div>
    <div class="actions">
        <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
        <form class="inline" method="POST" action="{{ url_for('delete_note', note_id=note['id']) }}" onsubmit="return confirm('Are you sure you want to delete this note?');">
            <button type="submit" class="btn-delete">Delete</button>
        </form>
    </div>
</div>
{% endblock %}
'''

from jinja2 import DictLoader

template_loader = DictLoader({
    'base': BASE_TEMPLATE,
    'index.html': INDEX_TEMPLATE,
    'create.html': CREATE_TEMPLATE,
    'edit.html': EDIT_TEMPLATE,
    'view.html': VIEW_TEMPLATE,
})

app.jinja_loader = template_loader


@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render_template_string(INDEX_TEMPLATE, notes=notes)


@app.route('/note/create', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            conn = get_db()
            conn.execute(
                'INSERT INTO notes (title, content) VALUES (?, ?)',
                (title, content)
            )
            conn.commit()
            conn.close()
            return redirect(url_for('index'))
    return render_template_string(CREATE_TEMPLATE)


@app.route('/note/<int:note_id>')
def view_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template_string(VIEW_TEMPLATE, note=note)


@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    if note is None:
        conn.close()
        return redirect(url_for('index'))
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
    return render_template_string(EDIT_TEMPLATE, note=note)


@app.route('/note/<int:note_id>/delete', methods=['POST'])
def delete_note(note_id):
    conn = get_db()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)