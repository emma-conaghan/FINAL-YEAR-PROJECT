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
            background-color: #4a90d9;
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
            background-color: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        header a:hover { background-color: rgba(255,255,255,0.3); }
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
            font-size: 1.3em;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .note-card .content {
            color: #555;
            margin-bottom: 12px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .note-card .meta {
            font-size: 0.8em;
            color: #999;
        }
        .note-card .actions {
            margin-top: 10px;
            display: flex;
            gap: 10px;
        }
        .note-card .actions a, .note-card .actions button {
            font-size: 0.85em;
            padding: 5px 12px;
            border-radius: 4px;
            text-decoration: none;
            cursor: pointer;
            border: none;
        }
        .btn-edit {
            background-color: #f0ad4e;
            color: white;
        }
        .btn-edit:hover { background-color: #ec971f; }
        .btn-delete {
            background-color: #d9534f;
            color: white;
        }
        .btn-delete:hover { background-color: #c9302c; }
        .btn-view {
            background-color: #5bc0de;
            color: white;
        }
        .btn-view:hover { background-color: #31b0d5; }
        form.note-form {
            background: white;
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        form.note-form label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #2c3e50;
        }
        form.note-form input[type="text"],
        form.note-form textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            margin-bottom: 15px;
            font-family: inherit;
        }
        form.note-form textarea {
            min-height: 200px;
            resize: vertical;
        }
        form.note-form input[type="text"]:focus,
        form.note-form textarea:focus {
            outline: none;
            border-color: #4a90d9;
            box-shadow: 0 0 0 2px rgba(74,144,217,0.2);
        }
        .btn-submit {
            background-color: #4a90d9;
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 1em;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn-submit:hover { background-color: #357abd; }
        .btn-back {
            display: inline-block;
            margin-bottom: 15px;
            color: #4a90d9;
            text-decoration: none;
            font-size: 0.9em;
        }
        .btn-back:hover { text-decoration: underline; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state p { font-size: 1.1em; margin-bottom: 15px; }
        .flash-message {
            background-color: #dff0d8;
            color: #3c763d;
            padding: 12px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            border: 1px solid #d6e9c6;
        }
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
            <div class="content">{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</div>
            <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
            <div class="actions">
                <a href="{{ url_for('view_note', note_id=note['id']) }}" class="btn-view">View</a>
                <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
                <form action="{{ url_for('delete_note', note_id=note['id']) }}" method="POST" style="display:inline;" onsubmit="return confirm('Delete this note?');">
                    <button type="submit" class="btn-delete">Delete</button>
                </form>
            </div>
        </div>
        {% endfor %}
    {% else %}
        <div class="empty-state">
            <p>No notes yet!</p>
            <a href="{{ url_for('create_note') }}" class="btn-submit" style="text-decoration:none; display:inline-block;">Create Your First Note</a>
        </div>
    {% endif %}
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="btn-back">← Back to Notes</a>
    <form class="note-form" method="POST" action="{{ url_for('create_note') }}">
        <label for="title">Title</label>
        <input type="text" id="title" name="title" placeholder="Enter note title..." required>
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here..."></textarea>
        <button type="submit" class="btn-submit">Save Note</button>
    </form>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="btn-back">← Back to Notes</a>
    <form class="note-form" method="POST" action="{{ url_for('edit_note', note_id=note['id']) }}">
        <label for="title">Title</label>
        <input type="text" id="title" name="title" value="{{ note['title'] }}" required>
        <label for="content">Content</label>
        <textarea id="content" name="content">{{ note['content'] }}</textarea>
        <button type="submit" class="btn-submit">Update Note</button>
    </form>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="btn-back">← Back to Notes</a>
    <div class="note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="content" style="margin-top: 15px;">{{ note['content'] }}</div>
        <div class="meta" style="margin-top: 15px;">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <div class="actions">
            <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
            <form action="{{ url_for('delete_note', note_id=note['id']) }}" method="POST" style="display:inline;" onsubmit="return confirm('Delete this note?');">
                <button type="submit" class="btn-delete">Delete</button>
            </form>
        </div>
    </div>
{% endblock %}
'''

from jinja2 import DictLoader

template_loader = DictLoader({
    'base': BASE_TEMPLATE,
    'index': INDEX_TEMPLATE,
    'create': CREATE_TEMPLATE,
    'edit': EDIT_TEMPLATE,
    'view': VIEW_TEMPLATE,
})

app.jinja_loader = template_loader


@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render_template_string(INDEX_TEMPLATE, notes=notes)


@app.route('/note/new', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title:
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
        if title:
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