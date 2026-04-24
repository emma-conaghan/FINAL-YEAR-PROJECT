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
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        header { background: #4a90d9; color: white; padding: 20px 0; margin-bottom: 30px; }
        header .container { display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 1.8em; }
        header a { color: white; text-decoration: none; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 5px; }
        header a:hover { background: rgba(255,255,255,0.3); }
        .note-card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: box-shadow 0.2s; }
        .note-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
        .note-card h2 { color: #4a90d9; margin-bottom: 10px; font-size: 1.3em; }
        .note-card p { color: #666; line-height: 1.6; white-space: pre-wrap; }
        .note-card .meta { font-size: 0.85em; color: #999; margin-top: 10px; }
        .note-card .actions { margin-top: 10px; }
        .note-card .actions a { text-decoration: none; margin-right: 10px; padding: 5px 12px; border-radius: 4px; font-size: 0.9em; }
        .btn-edit { background: #4a90d9; color: white; }
        .btn-edit:hover { background: #357abd; }
        .btn-delete { background: #e74c3c; color: white; }
        .btn-delete:hover { background: #c0392b; }
        .btn-view { background: #27ae60; color: white; }
        .btn-view:hover { background: #219a52; }
        form { background: white; border-radius: 8px; padding: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 5px; font-weight: 600; color: #555; }
        form input[type="text"], form textarea { width: 100%; padding: 10px; border: 2px solid #e0e0e0; border-radius: 5px; font-size: 1em; margin-bottom: 15px; font-family: inherit; }
        form input[type="text"]:focus, form textarea:focus { outline: none; border-color: #4a90d9; }
        form textarea { min-height: 200px; resize: vertical; }
        form button { background: #4a90d9; color: white; border: none; padding: 10px 25px; border-radius: 5px; font-size: 1em; cursor: pointer; }
        form button:hover { background: #357abd; }
        .back-link { display: inline-block; margin-bottom: 20px; color: #4a90d9; text-decoration: none; }
        .back-link:hover { text-decoration: underline; }
        .empty-state { text-align: center; padding: 40px; color: #999; }
        .empty-state p { font-size: 1.2em; margin-bottom: 15px; }
        .flash { background: #d4edda; color: #155724; padding: 10px 15px; border-radius: 5px; margin-bottom: 15px; }
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
            <a href="{{ url_for('delete_note', note_id=note['id']) }}" class="btn-delete" onclick="return confirm('Are you sure you want to delete this note?');">Delete</a>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="empty-state">
        <p>No notes yet!</p>
        <a href="{{ url_for('create_note') }}" class="btn-edit" style="padding: 10px 20px; font-size: 1em;">Create your first note</a>
    </div>
{% endif %}
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<a href="{{ url_for('index') }}" class="back-link">← Back to all notes</a>
<div class="note-card">
    <h2>{{ note['title'] }}</h2>
    <p>{{ note['content'] }}</p>
    <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
    <div class="actions" style="margin-top: 15px;">
        <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
        <a href="{{ url_for('delete_note', note_id=note['id']) }}" class="btn-delete" onclick="return confirm('Are you sure?');">Delete</a>
    </div>
</div>
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<a href="{{ url_for('index') }}" class="back-link">← Back to all notes</a>
<h2 style="margin-bottom: 20px;">Create New Note</h2>
<form method="POST">
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
<a href="{{ url_for('view_note', note_id=note['id']) }}" class="back-link">← Back to note</a>
<h2 style="margin-bottom: 20px;">Edit Note</h2>
<form method="POST">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" value="{{ note['title'] }}" required>
    <label for="content">Content</label>
    <textarea id="content" name="content" required>{{ note['content'] }}</textarea>
    <button type="submit">Update Note</button>
</form>
{% endblock %}
'''

from jinja2 import BaseLoader, TemplateNotFound, Environment

class DictLoader(BaseLoader):
    def __init__(self, templates):
        self.templates = templates

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)

templates_dict = {
    'base': BASE_TEMPLATE,
    'index': INDEX_TEMPLATE,
    'view': VIEW_TEMPLATE,
    'create': CREATE_TEMPLATE,
    'edit': EDIT_TEMPLATE,
}

app.jinja_loader = DictLoader(templates_dict)


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
    return render_template_string('{% extends "create" %}')


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
    return render_template_string(
        '{% extends "edit" %}',
        note=note
    )


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