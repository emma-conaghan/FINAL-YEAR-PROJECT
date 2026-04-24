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
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
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
        header h1 { font-size: 1.5rem; }
        header a {
            color: white;
            text-decoration: none;
            background-color: #3498db;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        header a:hover { background-color: #2980b9; }
        .note-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        .note-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
        .note-card h2 { font-size: 1.2rem; margin-bottom: 8px; color: #2c3e50; }
        .note-card p { color: #666; font-size: 0.95rem; white-space: pre-wrap; }
        .note-card .meta {
            margin-top: 12px;
            font-size: 0.8rem;
            color: #999;
        }
        .note-card .actions { margin-top: 10px; }
        .note-card .actions a {
            text-decoration: none;
            font-size: 0.85rem;
            margin-right: 12px;
            padding: 4px 10px;
            border-radius: 3px;
        }
        .note-card .actions .edit-btn { color: #3498db; border: 1px solid #3498db; }
        .note-card .actions .edit-btn:hover { background-color: #3498db; color: white; }
        .note-card .actions .delete-btn { color: #e74c3c; border: 1px solid #e74c3c; }
        .note-card .actions .delete-btn:hover { background-color: #e74c3c; color: white; }
        .note-card .actions .view-btn { color: #27ae60; border: 1px solid #27ae60; }
        .note-card .actions .view-btn:hover { background-color: #27ae60; color: white; }
        form { background: white; border-radius: 8px; padding: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 5px; font-weight: 600; color: #2c3e50; }
        form input[type="text"], form textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
            font-family: inherit;
            margin-bottom: 15px;
        }
        form input[type="text"]:focus, form textarea:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52,152,219,0.2);
        }
        form textarea { min-height: 200px; resize: vertical; }
        form button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
        }
        form button:hover { background-color: #2980b9; }
        .back-link { display: inline-block; margin-bottom: 15px; color: #3498db; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        .empty-state { text-align: center; padding: 60px 20px; color: #999; }
        .empty-state p { font-size: 1.1rem; margin-bottom: 15px; }
        .view-note { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .view-note h2 { font-size: 1.5rem; color: #2c3e50; margin-bottom: 15px; }
        .view-note .content { white-space: pre-wrap; color: #555; line-height: 1.8; }
        .view-note .meta { margin-top: 20px; font-size: 0.85rem; color: #999; }
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
            <p>{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</p>
            <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
            <div class="actions">
                <a href="{{ url_for('view_note', note_id=note['id']) }}" class="view-btn">View</a>
                <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="edit-btn">Edit</a>
                <a href="{{ url_for('delete_note', note_id=note['id']) }}" class="delete-btn" onclick="return confirm('Are you sure you want to delete this note?');">Delete</a>
            </div>
        </div>
        {% endfor %}
    {% else %}
        <div class="empty-state">
            <p>No notes yet!</p>
            <a href="{{ url_for('create_note') }}">Create your first note</a>
        </div>
    {% endif %}
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
    <form method="POST">
        <h2 style="margin-bottom: 20px; color: #2c3e50;">Create New Note</h2>
        <label for="title">Title</label>
        <input type="text" id="title" name="title" placeholder="Enter note title..." required>
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here..." required></textarea>
        <button type="submit">Save Note</button>
    </form>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
    <form method="POST">
        <h2 style="margin-bottom: 20px; color: #2c3e50;">Edit Note</h2>
        <label for="title">Title</label>
        <input type="text" id="title" name="title" value="{{ note['title'] }}" required>
        <label for="content">Content</label>
        <textarea id="content" name="content" required>{{ note['content'] }}</textarea>
        <button type="submit">Update Note</button>
    </form>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
    <div class="view-note">
        <h2>{{ note['title'] }}</h2>
        <div class="content">{{ note['content'] }}</div>
        <div class="meta">
            Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}
        </div>
        <div style="margin-top: 15px;">
            <a href="{{ url_for('edit_note', note_id=note['id']) }}" style="color: #3498db; text-decoration: none;">Edit this note</a>
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


@app.route('/create', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            conn = get_db()
            conn.execute('INSERT INTO notes (title, content) VALUES (?, ?)', (title, content))
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


@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    conn = get_db()
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
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template_string(EDIT_TEMPLATE, note=note)


@app.route('/delete/<int:note_id>')
def delete_note(note_id):
    conn = get_db()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)