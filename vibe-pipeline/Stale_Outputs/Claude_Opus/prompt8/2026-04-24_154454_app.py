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
        header h1 { font-size: 1.5rem; }
        header a {
            color: white;
            text-decoration: none;
            background-color: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 4px;
        }
        header a:hover { background-color: rgba(255,255,255,0.3); }
        .note-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        .note-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
        .note-card h2 { font-size: 1.2rem; margin-bottom: 8px; color: #4a90d9; }
        .note-card .content { color: #666; margin-bottom: 12px; white-space: pre-wrap; }
        .note-card .meta { font-size: 0.8rem; color: #999; }
        .note-card .actions { margin-top: 10px; }
        .note-card .actions a, .note-card .actions button {
            font-size: 0.85rem;
            margin-right: 10px;
            text-decoration: none;
        }
        .note-card .actions a { color: #4a90d9; }
        .note-card .actions a:hover { text-decoration: underline; }
        .note-card .actions .delete-btn {
            color: #e74c3c;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 0.85rem;
            padding: 0;
        }
        .note-card .actions .delete-btn:hover { text-decoration: underline; }
        form { background: white; border-radius: 8px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 6px; font-weight: 600; }
        form input[type="text"], form textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
            font-family: inherit;
            margin-bottom: 16px;
        }
        form input[type="text"]:focus, form textarea:focus {
            outline: none;
            border-color: #4a90d9;
            box-shadow: 0 0 0 2px rgba(74,144,217,0.2);
        }
        form textarea { min-height: 200px; resize: vertical; }
        .btn {
            background-color: #4a90d9;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
        }
        .btn:hover { background-color: #357abd; }
        .btn-secondary {
            background-color: #95a5a6;
            margin-left: 10px;
            text-decoration: none;
            display: inline-block;
            padding: 10px 24px;
            border-radius: 4px;
            color: white;
            font-size: 1rem;
        }
        .btn-secondary:hover { background-color: #7f8c8d; }
        .empty-state { text-align: center; padding: 60px 20px; color: #999; }
        .empty-state h2 { margin-bottom: 10px; }
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
            <div class="content">{{ note['content'][:300] }}{% if note['content']|length > 300 %}...{% endif %}</div>
            <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
            <div class="actions">
                <a href="{{ url_for('view_note', note_id=note['id']) }}">View</a>
                <a href="{{ url_for('edit_note', note_id=note['id']) }}">Edit</a>
                <form method="POST" action="{{ url_for('delete_note', note_id=note['id']) }}" style="display:inline;">
                    <button type="submit" class="delete-btn" onclick="return confirm('Delete this note?')">Delete</button>
                </form>
            </div>
        </div>
        {% endfor %}
    {% else %}
        <div class="empty-state">
            <h2>No notes yet</h2>
            <p>Click "+ New Note" to create your first note.</p>
        </div>
    {% endif %}
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <div class="note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="content">{{ note['content'] }}</div>
        <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <div class="actions" style="margin-top: 16px;">
            <a href="{{ url_for('edit_note', note_id=note['id']) }}">Edit</a>
            <a href="{{ url_for('index') }}">Back to all notes</a>
        </div>
    </div>
{% endblock %}
'''

FORM_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <form method="POST">
        <h2 style="margin-bottom: 20px;">{{ 'Edit Note' if note else 'New Note' }}</h2>
        <label for="title">Title</label>
        <input type="text" id="title" name="title" value="{{ note['title'] if note else '' }}" required placeholder="Enter note title...">
        <label for="content">Content</label>
        <textarea id="content" name="content" required placeholder="Write your note here...">{{ note['content'] if note else '' }}</textarea>
        <button type="submit" class="btn">{{ 'Update' if note else 'Save' }}</button>
        <a href="{{ url_for('index') }}" class="btn-secondary">Cancel</a>
    </form>
{% endblock %}
'''

from jinja2 import DictLoader

template_loader = DictLoader({
    'base': BASE_TEMPLATE,
    'index': INDEX_TEMPLATE,
    'view': VIEW_TEMPLATE,
    'form': FORM_TEMPLATE,
})

app.jinja_loader = template_loader


@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render_template_string(
        '{% extends "index" %}',
        notes=notes
    )


@app.route('/note/<int:note_id>')
def view_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template_string(
        '{% extends "view" %}',
        note=note
    )


@app.route('/create', methods=['GET', 'POST'])
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
    return render_template_string(
        '{% extends "form" %}',
        note=None
    )


@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
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
    return render_template_string(
        '{% extends "form" %}',
        note=note
    )


@app.route('/delete/<int:note_id>', methods=['POST'])
def delete_note(note_id):
    conn = get_db()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)