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
        * { margin: 0; padding: 0; box-sizing: border-box; }
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
        header h1 a { color: white; text-decoration: none; }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover { background-color: #2980b9; }
        .btn-success { background-color: #27ae60; }
        .btn-success:hover { background-color: #219a52; }
        .btn-danger { background-color: #e74c3c; }
        .btn-danger:hover { background-color: #c0392b; }
        .btn-secondary { background-color: #95a5a6; }
        .btn-secondary:hover { background-color: #7f8c8d; }
        .note-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        .note-card:hover { box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
        .note-card h2 { margin-bottom: 10px; color: #2c3e50; }
        .note-card .note-preview {
            color: #666;
            margin-bottom: 15px;
            white-space: pre-wrap;
            max-height: 100px;
            overflow: hidden;
        }
        .note-card .note-meta {
            font-size: 12px;
            color: #999;
            margin-bottom: 10px;
        }
        .note-card .actions { display: flex; gap: 10px; }
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
            font-size: 16px;
            font-family: inherit;
            transition: border-color 0.2s;
        }
        .form-group input[type="text"]:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #3498db;
        }
        .form-group textarea { min-height: 300px; resize: vertical; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state h2 { margin-bottom: 10px; }
        .note-full-content {
            white-space: pre-wrap;
            line-height: 1.8;
            color: #444;
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .note-header {
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .note-header h2 { color: #2c3e50; margin-bottom: 10px; }
        .note-header .note-meta { color: #999; font-size: 13px; }
        .note-header .actions { margin-top: 15px; display: flex; gap: 10px; }
        .form-card {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-actions { display: flex; gap: 10px; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="/">📝 Personal Notes</a></h1>
            <a href="/notes/new" class="btn btn-success">+ New Note</a>
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
            <div class="note-meta">{{ note['updated_at'] }}</div>
            <h2><a href="/notes/{{ note['id'] }}" style="text-decoration:none;color:#2c3e50;">{{ note['title'] | e }}</a></h2>
            <div class="note-preview">{{ note['content'][:200] | e }}{% if note['content']|length > 200 %}...{% endif %}</div>
            <div class="actions">
                <a href="/notes/{{ note['id'] }}" class="btn">View</a>
                <a href="/notes/{{ note['id'] }}/edit" class="btn btn-secondary">Edit</a>
                <form action="/notes/{{ note['id'] }}/delete" method="POST" style="display:inline;" onsubmit="return confirm('Are you sure you want to delete this note?');">
                    <button type="submit" class="btn btn-danger">Delete</button>
                </form>
            </div>
        </div>
        {% endfor %}
    {% else %}
        <div class="empty-state">
            <h2>No notes yet</h2>
            <p>Click the "+ New Note" button to create your first note.</p>
        </div>
    {% endif %}
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <div class="note-header">
        <h2>{{ note['title'] | e }}</h2>
        <div class="note-meta">
            Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}
        </div>
        <div class="actions">
            <a href="/notes/{{ note['id'] }}/edit" class="btn btn-secondary">Edit</a>
            <form action="/notes/{{ note['id'] }}/delete" method="POST" style="display:inline;" onsubmit="return confirm('Are you sure?');">
                <button type="submit" class="btn btn-danger">Delete</button>
            </form>
            <a href="/" class="btn">Back to Notes</a>
        </div>
    </div>
    <div class="note-full-content">{{ note['content'] | e }}</div>
{% endblock %}
'''

FORM_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <div class="form-card">
        <h2 style="margin-bottom:20px;color:#2c3e50;">{{ form_title }}</h2>
        <form method="POST">
            <div class="form-group">
                <label for="title">Title</label>
                <input type="text" id="title" name="title" value="{{ note_title | default('') | e }}" placeholder="Enter note title..." required>
            </div>
            <div class="form-group">
                <label for="content">Content</label>
                <textarea id="content" name="content" placeholder="Write your note here..." required>{{ note_content | default('') | e }}</textarea>
            </div>
            <div class="form-actions">
                <button type="submit" class="btn btn-success">{{ submit_text }}</button>
                <a href="/" class="btn btn-secondary">Cancel</a>
            </div>
        </form>
    </div>
{% endblock %}
'''

from jinja2 import DictLoader, Environment

jinja_env = Environment(
    loader=DictLoader({
        'base': BASE_TEMPLATE,
        'index': INDEX_TEMPLATE,
        'view': VIEW_TEMPLATE,
        'form': FORM_TEMPLATE,
    }),
    autoescape=True
)

def render(template_name, **kwargs):
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)


@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render('index', notes=notes)


@app.route('/notes/new', methods=['GET', 'POST'])
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
    return render('form', form_title='Create New Note', submit_text='Create Note')


@app.route('/notes/<int:note_id>')
def view_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render('view', note=note)


@app.route('/notes/<int:note_id>/edit', methods=['GET', 'POST'])
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
    return render('form',
                  form_title='Edit Note',
                  submit_text='Save Changes',
                  note_title=note['title'],
                  note_content=note['content'])


@app.route('/notes/<int:note_id>/delete', methods=['POST'])
def delete_note(note_id):
    conn = get_db()
    conn.execute('DELETE FROM notes WHERE id = ?', (note_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)