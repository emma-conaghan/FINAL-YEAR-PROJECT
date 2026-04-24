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
        header h1 { font-size: 1.5rem; }
        header a {
            color: white;
            text-decoration: none;
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        header a:hover { background: rgba(255,255,255,0.3); }
        .note-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        .note-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
        .note-card h2 {
            font-size: 1.2rem;
            margin-bottom: 8px;
            color: #2c3e50;
        }
        .note-card p {
            color: #666;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .note-card .meta {
            font-size: 0.8rem;
            color: #999;
            margin-top: 12px;
        }
        .note-card .actions {
            margin-top: 12px;
            display: flex;
            gap: 10px;
        }
        .note-card .actions a, .note-card .actions button {
            font-size: 0.85rem;
            padding: 6px 12px;
            border-radius: 4px;
            text-decoration: none;
            cursor: pointer;
            border: none;
        }
        .btn-edit { background: #4a90d9; color: white; }
        .btn-edit:hover { background: #357abd; }
        .btn-delete { background: #e74c3c; color: white; }
        .btn-delete:hover { background: #c0392b; }
        .btn-view { background: #27ae60; color: white; }
        .btn-view:hover { background: #219a52; }
        form { background: white; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        form label { display: block; font-weight: 600; margin-bottom: 6px; color: #2c3e50; }
        form input[type="text"], form textarea {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 1rem;
            font-family: inherit;
            margin-bottom: 16px;
            transition: border-color 0.2s;
        }
        form input[type="text"]:focus, form textarea:focus {
            outline: none;
            border-color: #4a90d9;
        }
        form textarea { min-height: 200px; resize: vertical; }
        form button {
            background: #4a90d9;
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 1rem;
            border-radius: 6px;
            cursor: pointer;
        }
        form button:hover { background: #357abd; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state h2 { margin-bottom: 10px; color: #bbb; }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #4a90d9;
            text-decoration: none;
        }
        .back-link:hover { text-decoration: underline; }
        .flash {
            background: #d4edda;
            color: #155724;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            border: 1px solid #c3e6cb;
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
            <p>{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</p>
            <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
            <div class="actions">
                <a href="{{ url_for('view_note', note_id=note['id']) }}" class="btn-view">View</a>
                <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
                <form action="{{ url_for('delete_note', note_id=note['id']) }}" method="POST" style="display:inline; background:none; padding:0; box-shadow:none;" onsubmit="return confirm('Delete this note?');">
                    <button type="submit" class="btn-delete">Delete</button>
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

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
    <form method="POST">
        <h2 style="margin-bottom: 20px; color: #2c3e50;">Create New Note</h2>
        <label for="title">Title</label>
        <input type="text" id="title" name="title" placeholder="Enter note title..." required value="{{ title or '' }}">
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here..." required>{{ content or '' }}</textarea>
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
        <input type="text" id="title" name="title" placeholder="Enter note title..." required value="{{ note['title'] }}">
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here..." required>{{ note['content'] }}</textarea>
        <button type="submit">Update Note</button>
    </form>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
    <div class="note-card">
        <h2 style="font-size: 1.5rem;">{{ note['title'] }}</h2>
        <div class="meta" style="margin-bottom: 16px;">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <p style="font-size: 1rem; line-height: 1.8;">{{ note['content'] }}</p>
        <div class="actions" style="margin-top: 20px;">
            <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
            <form action="{{ url_for('delete_note', note_id=note['id']) }}" method="POST" style="display:inline; background:none; padding:0; box-shadow:none;" onsubmit="return confirm('Delete this note?');">
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


@app.route('/note/new', methods=['GET', 'POST'])
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
        return render_template_string(CREATE_TEMPLATE, title=title, content=content)
    return render_template_string(CREATE_TEMPLATE, title='', content='')


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