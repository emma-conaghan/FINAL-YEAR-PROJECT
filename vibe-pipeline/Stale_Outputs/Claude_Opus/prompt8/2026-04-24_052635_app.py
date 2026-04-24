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
        .note-card .content { color: #555; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; }
        .note-card .meta { font-size: 0.85em; color: #999; margin-top: 12px; }
        .note-card .actions { margin-top: 12px; }
        .note-card .actions a, .note-card .actions button { 
            display: inline-block; padding: 6px 12px; margin-right: 8px; border-radius: 4px; 
            text-decoration: none; font-size: 0.9em; cursor: pointer; border: none;
        }
        .btn-edit { background: #4a90d9; color: white; }
        .btn-edit:hover { background: #357abd; }
        .btn-delete { background: #e74c3c; color: white; }
        .btn-delete:hover { background: #c0392b; }
        .btn-view { background: #27ae60; color: white; }
        .btn-view:hover { background: #219a52; }
        form { background: white; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 6px; font-weight: 600; color: #555; }
        form input[type="text"], form textarea { 
            width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; 
            font-size: 1em; font-family: inherit; margin-bottom: 16px;
        }
        form input[type="text"]:focus, form textarea:focus { outline: none; border-color: #4a90d9; box-shadow: 0 0 0 2px rgba(74,144,217,0.2); }
        form textarea { min-height: 200px; resize: vertical; }
        form button { background: #4a90d9; color: white; border: none; padding: 10px 24px; border-radius: 4px; font-size: 1em; cursor: pointer; }
        form button:hover { background: #357abd; }
        .empty-state { text-align: center; padding: 60px 20px; color: #999; }
        .empty-state h2 { margin-bottom: 10px; }
        .flash { background: #d4edda; color: #155724; padding: 12px 20px; border-radius: 4px; margin-bottom: 20px; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>📝 Personal Notes</h1>
            <a href="{{ url_for('index') }}">All Notes</a>
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
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <h2>Your Notes ({{ notes|length }})</h2>
    <a href="{{ url_for('create') }}" class="btn-edit" style="padding: 10px 20px; border-radius: 4px; text-decoration: none; color: white;">+ New Note</a>
</div>
{% if message %}
<div class="flash">{{ message }}</div>
{% endif %}
{% if notes %}
    {% for note in notes %}
    <div class="note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="content">{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</div>
        <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <div class="actions">
            <a href="{{ url_for('view', note_id=note['id']) }}" class="btn-view">View</a>
            <a href="{{ url_for('edit', note_id=note['id']) }}" class="btn-edit">Edit</a>
            <form method="POST" action="{{ url_for('delete', note_id=note['id']) }}" style="display: inline;">
                <button type="submit" class="btn-delete" onclick="return confirm('Are you sure you want to delete this note?');">Delete</button>
            </form>
        </div>
    </div>
    {% endfor %}
{% else %}
    <div class="empty-state">
        <h2>No notes yet!</h2>
        <p>Click "+ New Note" to create your first note.</p>
    </div>
{% endif %}
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h2 style="margin-bottom: 20px;">Create New Note</h2>
{% if error %}
<div class="flash" style="background: #f8d7da; color: #721c24; border-color: #f5c6cb;">{{ error }}</div>
{% endif %}
<form method="POST">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" placeholder="Enter note title..." value="{{ title or '' }}" required>
    <label for="content">Content</label>
    <textarea id="content" name="content" placeholder="Write your note here...">{{ content or '' }}</textarea>
    <button type="submit">Save Note</button>
    <a href="{{ url_for('index') }}" style="margin-left: 12px; color: #999;">Cancel</a>
</form>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h2 style="margin-bottom: 20px;">Edit Note</h2>
{% if error %}
<div class="flash" style="background: #f8d7da; color: #721c24; border-color: #f5c6cb;">{{ error }}</div>
{% endif %}
<form method="POST">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" placeholder="Enter note title..." value="{{ note['title'] }}" required>
    <label for="content">Content</label>
    <textarea id="content" name="content" placeholder="Write your note here...">{{ note['content'] }}</textarea>
    <button type="submit">Update Note</button>
    <a href="{{ url_for('view', note_id=note['id']) }}" style="margin-left: 12px; color: #999;">Cancel</a>
</form>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="note-card">
    <h2>{{ note['title'] }}</h2>
    <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
    <hr style="margin: 16px 0; border: none; border-top: 1px solid #eee;">
    <div class="content">{{ note['content'] }}</div>
    <div class="actions" style="margin-top: 20px;">
        <a href="{{ url_for('edit', note_id=note['id']) }}" class="btn-edit">Edit</a>
        <form method="POST" action="{{ url_for('delete', note_id=note['id']) }}" style="display: inline;">
            <button type="submit" class="btn-delete" onclick="return confirm('Are you sure you want to delete this note?');">Delete</button>
        </form>
        <a href="{{ url_for('index') }}" style="margin-left: 8px; color: #999;">Back to all notes</a>
    </div>
</div>
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

templates = {
    'base': BASE_TEMPLATE,
    'index': INDEX_TEMPLATE,
    'create': CREATE_TEMPLATE,
    'edit': EDIT_TEMPLATE,
    'view': VIEW_TEMPLATE,
}

app.jinja_loader = DictLoader(templates)


@app.route('/')
def index():
    message = request.args.get('message', '')
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render_template_string(
        '{% extends "index" %}',
        notes=notes,
        message=message
    )


@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if not title:
            return render_template_string(
                '{% extends "create" %}',
                error='Title is required.',
                title=title,
                content=content
            )
        conn = get_db()
        conn.execute(
            'INSERT INTO notes (title, content) VALUES (?, ?)',
            (title, content)
        )
        conn.commit()
        conn.close()
        return redirect(url_for('index', message='Note created successfully!'))
    return render_template_string('{% extends "create" %}', error=None, title='', content='')


@app.route('/note/<int:note_id>')
def view(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index', message='Note not found.'))
    return render_template_string('{% extends "view" %}', note=note)


@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    if note is None:
        conn.close()
        return redirect(url_for('index', message='Note not found.'))
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if not title:
            conn.close()
            note_dict = dict(note)
            note_dict['title'] = title
            note_dict['content'] = content
            return render_template_string(
                '{% extends "edit" %}',
                error='Title is required.',
                note=note_dict
            )
        conn.execute(
            'UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
            (title, content, note_id)
        )
        conn.commit()
        conn.close()
        return redirect(url_for('view', note_id=note_id))
    conn.close()
    return render_template_string('{% extends "edit" %}', error=None, note=note)


@app.route('/note/<int:note_id>/delete', methods=['POST'])
def delete(note_id):
    conn = get_db()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index', message='Note deleted successfully!'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)