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
        header { background: #4a90d9; color: white; padding: 20px 0; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        header .container { display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 1.5em; }
        header a { color: white; text-decoration: none; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 4px; }
        header a:hover { background: rgba(255,255,255,0.3); }
        .note-card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: box-shadow 0.2s; }
        .note-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
        .note-card h2 { margin-bottom: 8px; color: #4a90d9; }
        .note-card .content { color: #555; line-height: 1.6; white-space: pre-wrap; margin-bottom: 12px; }
        .note-card .meta { font-size: 0.85em; color: #999; }
        .note-card .actions { margin-top: 10px; }
        .note-card .actions a, .note-card .actions button { margin-right: 10px; text-decoration: none; font-size: 0.9em; }
        .note-card .actions a { color: #4a90d9; }
        .note-card .actions a:hover { text-decoration: underline; }
        .note-card .actions .delete-btn { background: none; border: none; color: #e74c3c; cursor: pointer; font-size: 0.9em; padding: 0; }
        .note-card .actions .delete-btn:hover { text-decoration: underline; }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; margin-bottom: 6px; font-weight: 600; color: #555; }
        .form-group input[type="text"], .form-group textarea {
            width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 1em; font-family: inherit;
        }
        .form-group input[type="text"]:focus, .form-group textarea:focus { outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }
        .form-group textarea { min-height: 200px; resize: vertical; }
        .btn { display: inline-block; padding: 10px 24px; background: #4a90d9; color: white; border: none; border-radius: 4px; font-size: 1em; cursor: pointer; text-decoration: none; }
        .btn:hover { background: #357abd; }
        .btn-secondary { background: #95a5a6; }
        .btn-secondary:hover { background: #7f8c8d; }
        .empty-state { text-align: center; padding: 60px 20px; color: #999; }
        .empty-state h2 { margin-bottom: 10px; }
        .flash { padding: 12px 20px; margin-bottom: 20px; border-radius: 4px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
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
        <div class="content">{{ note['content'] }}</div>
        <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <div class="actions">
            <a href="{{ url_for('view_note', note_id=note['id']) }}">View</a>
            <a href="{{ url_for('edit_note', note_id=note['id']) }}">Edit</a>
            <form method="POST" action="{{ url_for('delete_note', note_id=note['id']) }}" style="display:inline;" onsubmit="return confirm('Delete this note?');">
                <button type="submit" class="delete-btn">Delete</button>
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
    <div class="actions" style="margin-top:16px;">
        <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn" style="font-size:0.9em; padding:6px 16px;">Edit</a>
        <a href="{{ url_for('index') }}" class="btn btn-secondary" style="font-size:0.9em; padding:6px 16px;">Back</a>
    </div>
</div>
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="note-card">
    <h2>Create New Note</h2>
    <form method="POST" style="margin-top:16px;">
        <div class="form-group">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" required placeholder="Enter note title...">
        </div>
        <div class="form-group">
            <label for="content">Content</label>
            <textarea id="content" name="content" required placeholder="Write your note here..."></textarea>
        </div>
        <button type="submit" class="btn">Save Note</button>
        <a href="{{ url_for('index') }}" class="btn btn-secondary" style="margin-left:8px;">Cancel</a>
    </form>
</div>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="note-card">
    <h2>Edit Note</h2>
    <form method="POST" style="margin-top:16px;">
        <div class="form-group">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" required value="{{ note['title'] }}">
        </div>
        <div class="form-group">
            <label for="content">Content</label>
            <textarea id="content" name="content" required>{{ note['content'] }}</textarea>
        </div>
        <button type="submit" class="btn">Update Note</button>
        <a href="{{ url_for('view_note', note_id=note['id']) }}" class="btn btn-secondary" style="margin-left:8px;">Cancel</a>
    </form>
</div>
{% endblock %}
'''

@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render_template_string(INDEX_TEMPLATE, notes=notes, base=BASE_TEMPLATE)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template_string(VIEW_TEMPLATE, note=note)

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

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    conn = get_db()
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            conn.execute('UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (title, content, note_id))
            conn.commit()
            conn.close()
            return redirect(url_for('view_note', note_id=note_id))
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template_string(EDIT_TEMPLATE, note=note)

@app.route('/delete/<int:note_id>', methods=['POST'])
def delete_note(note_id):
    conn = get_db()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

# Override Jinja2 template loading to support extends with inline templates
from jinja2 import BaseLoader, TemplateNotFound

class InlineLoader(BaseLoader):
    def __init__(self, templates):
        self.templates = templates

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)

TEMPLATES = {
    'base': BASE_TEMPLATE,
    'index': INDEX_TEMPLATE,
    'view': VIEW_TEMPLATE,
    'create': CREATE_TEMPLATE,
    'edit': EDIT_TEMPLATE,
}

app.jinja_loader = InlineLoader(TEMPLATES)

# Override routes to use render_template instead of render_template_string
from flask import render_template

@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render_template('index', notes=notes)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template('view', note=note)

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
    return render_template('create')

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    conn = get_db()
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            conn.execute('UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?', (title, content, note_id))
            conn.commit()
            conn.close()
            return redirect(url_for('view_note', note_id=note_id))
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template('edit', note=note)

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