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
            color: #666;
            margin-bottom: 12px;
            white-space: pre-wrap;
        }
        .note-card .meta {
            font-size: 0.85em;
            color: #999;
        }
        .note-card .actions { margin-top: 10px; }
        .note-card .actions a, .note-card .actions button {
            display: inline-block;
            padding: 6px 14px;
            margin-right: 8px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.9em;
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
        form {
            background: white;
            border-radius: 8px;
            padding: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        form label {
            display: block;
            font-weight: 600;
            margin-bottom: 5px;
            margin-top: 15px;
        }
        form label:first-child { margin-top: 0; }
        form input[type="text"], form textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            font-family: inherit;
        }
        form textarea { min-height: 200px; resize: vertical; }
        form input[type="text"]:focus, form textarea:focus {
            outline: none;
            border-color: #4a90d9;
            box-shadow: 0 0 0 2px rgba(74,144,217,0.2);
        }
        .btn-submit {
            margin-top: 20px;
            background-color: #4a90d9;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 1em;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn-submit:hover { background-color: #357abd; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state h2 { margin-bottom: 10px; }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #4a90d9;
            text-decoration: none;
        }
        .back-link:hover { text-decoration: underline; }
        .flash-message {
            background-color: #dff0d8;
            color: #3c763d;
            padding: 12px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
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
                <form method="POST" action="{{ url_for('delete_note', note_id=note['id']) }}" style="display:inline; background:none; padding:0; box-shadow:none;">
                    <button type="submit" class="btn-delete" onclick="return confirm('Are you sure you want to delete this note?')">Delete</button>
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
    <form method="POST" action="{{ url_for('create_note') }}">
        <h2>Create New Note</h2>
        <label for="title">Title</label>
        <input type="text" id="title" name="title" placeholder="Enter note title..." required value="{{ title or '' }}">
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here...">{{ content or '' }}</textarea>
        <button type="submit" class="btn-submit">Save Note</button>
    </form>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
    <form method="POST" action="{{ url_for('edit_note', note_id=note['id']) }}">
        <h2>Edit Note</h2>
        <label for="title">Title</label>
        <input type="text" id="title" name="title" placeholder="Enter note title..." required value="{{ note['title'] }}">
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here...">{{ note['content'] }}</textarea>
        <button type="submit" class="btn-submit">Update Note</button>
    </form>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
    <div class="note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="content" style="white-space: pre-wrap;">{{ note['content'] }}</div>
        <div class="meta">Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</div>
        <div class="actions">
            <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
            <form method="POST" action="{{ url_for('delete_note', note_id=note['id']) }}" style="display:inline; background:none; padding:0; box-shadow:none;">
                <button type="submit" class="btn-delete" onclick="return confirm('Are you sure you want to delete this note?')">Delete</button>
            </form>
        </div>
    </div>
{% endblock %}
'''

class TemplateLoader:
    templates = {
        'base': BASE_TEMPLATE,
        'index': INDEX_TEMPLATE,
        'create': CREATE_TEMPLATE,
        'edit': EDIT_TEMPLATE,
        'view': VIEW_TEMPLATE,
    }

def render(template_name, **kwargs):
    from jinja2 import Environment, BaseLoader, TemplateNotFound

    class DictLoader(BaseLoader):
        def __init__(self, templates):
            self.templates = templates

        def get_source(self, environment, template):
            if template in self.templates:
                source = self.templates[template]
                return source, template, lambda: True
            raise TemplateNotFound(template)

    env = Environment(loader=DictLoader(TemplateLoader.templates))
    env.globals['url_for'] = url_for
    template = env.get_template(template_name)
    return template.render(**kwargs)


@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render('index', notes=notes)


@app.route('/notes/create', methods=['GET', 'POST'])
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
        return render('create', title=title, content=content)
    return render('create', title='', content='')


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
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render('edit', note=note)


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