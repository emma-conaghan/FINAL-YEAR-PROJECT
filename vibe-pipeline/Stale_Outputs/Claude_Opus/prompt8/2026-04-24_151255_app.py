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
        header h1 { font-size: 1.5em; }
        header a {
            color: white;
            text-decoration: none;
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 5px;
            transition: background 0.2s;
        }
        header a:hover { background: rgba(255,255,255,0.3); }
        .note-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        .note-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
        .note-card h2 { color: #4a90d9; margin-bottom: 8px; font-size: 1.2em; }
        .note-card .content {
            color: #555;
            white-space: pre-wrap;
            word-wrap: break-word;
            margin-bottom: 12px;
        }
        .note-card .meta {
            font-size: 0.8em;
            color: #999;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .note-card .actions a {
            text-decoration: none;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            margin-left: 8px;
        }
        .btn-edit { background: #e8f0fe; color: #4a90d9; }
        .btn-edit:hover { background: #d0e1fd; }
        .btn-delete { background: #fde8e8; color: #d94a4a; }
        .btn-delete:hover { background: #fcd0d0; }
        form { background: white; border-radius: 8px; padding: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        form h2 { margin-bottom: 20px; color: #4a90d9; }
        label { display: block; margin-bottom: 5px; font-weight: 600; color: #555; }
        input[type="text"], textarea {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 1em;
            font-family: inherit;
            margin-bottom: 15px;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: #4a90d9;
        }
        textarea { min-height: 200px; resize: vertical; }
        button[type="submit"] {
            background: #4a90d9;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: background 0.2s;
        }
        button[type="submit"]:hover { background: #3a7bc8; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state h2 { margin-bottom: 10px; color: #bbb; }
        .flash { background: #d4edda; color: #155724; padding: 12px 20px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>📝 Personal Notes</h1>
            <nav>
                <a href="/">All Notes</a>
                <a href="/new">+ New Note</a>
            </nav>
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
{% if message %}
<div class="flash">{{ message }}</div>
{% endif %}
{% if notes %}
    {% for note in notes %}
    <div class="note-card">
        <h2><a href="/note/{{ note.id }}" style="text-decoration:none;color:#4a90d9;">{{ note.title }}</a></h2>
        <div class="content">{{ note.content[:200] }}{% if note.content|length > 200 %}...{% endif %}</div>
        <div class="meta">
            <span>{{ note.updated_at }}</span>
            <div class="actions">
                <a href="/edit/{{ note.id }}" class="btn-edit">Edit</a>
                <a href="/delete/{{ note.id }}" class="btn-delete" onclick="return confirm('Delete this note?');">Delete</a>
            </div>
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
    <h2>{{ note.title }}</h2>
    <div class="content" style="margin-top:15px;">{{ note.content }}</div>
    <div class="meta" style="margin-top:20px;">
        <span>Created: {{ note.created_at }} | Updated: {{ note.updated_at }}</span>
        <div class="actions">
            <a href="/edit/{{ note.id }}" class="btn-edit">Edit</a>
            <a href="/delete/{{ note.id }}" class="btn-delete" onclick="return confirm('Delete this note?');">Delete</a>
        </div>
    </div>
</div>
{% endblock %}
'''

FORM_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<form method="POST">
    <h2>{{ 'Edit Note' if note else 'New Note' }}</h2>
    <label for="title">Title</label>
    <input type="text" id="title" name="title" value="{{ note.title if note else '' }}" required placeholder="Enter note title...">
    <label for="content">Content</label>
    <textarea id="content" name="content" required placeholder="Write your note here...">{{ note.content if note else '' }}</textarea>
    <button type="submit">{{ 'Update Note' if note else 'Save Note' }}</button>
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
    'form': FORM_TEMPLATE,
}

jinja_env = Environment(loader=DictLoader(templates_dict))

def render(template_name, **kwargs):
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)

@app.route('/')
def index():
    message = request.args.get('message', '')
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render('index', notes=notes, message=message)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index', message='Note not found.'))
    return render('view', note=note)

@app.route('/new', methods=['GET', 'POST'])
def new_note():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            conn = get_db()
            conn.execute('INSERT INTO notes (title, content) VALUES (?, ?)', (title, content))
            conn.commit()
            conn.close()
            return redirect(url_for('index', message='Note created successfully!'))
    return render('form', note=None)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    if note is None:
        conn.close()
        return redirect(url_for('index', message='Note not found.'))
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
            return redirect(url_for('index', message='Note updated successfully!'))
    conn.close()
    return render('form', note=note)

@app.route('/delete/<int:note_id>')
def delete_note(note_id):
    conn = get_db()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index', message='Note deleted.'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)