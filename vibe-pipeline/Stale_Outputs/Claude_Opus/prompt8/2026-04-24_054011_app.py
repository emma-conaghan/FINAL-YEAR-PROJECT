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
        header h1 { font-size: 1.8em; }
        header a {
            color: white;
            text-decoration: none;
            background-color: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 5px;
            font-weight: bold;
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
        .note-card h2 { margin-bottom: 10px; color: #4a90d9; }
        .note-card .content {
            color: #666;
            white-space: pre-wrap;
            margin-bottom: 15px;
        }
        .note-card .meta {
            font-size: 0.85em;
            color: #999;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .note-card .actions a {
            text-decoration: none;
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 0.9em;
            margin-left: 8px;
        }
        .btn-edit { background-color: #f0ad4e; color: white; }
        .btn-edit:hover { background-color: #ec971f; }
        .btn-delete { background-color: #d9534f; color: white; }
        .btn-delete:hover { background-color: #c9302c; }
        .btn-view { background-color: #5bc0de; color: white; }
        .btn-view:hover { background-color: #31b0d5; }
        form { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        form label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        form input[type="text"], form textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            margin-bottom: 20px;
            font-family: inherit;
        }
        form input[type="text"]:focus, form textarea:focus {
            outline: none;
            border-color: #4a90d9;
        }
        form textarea { min-height: 200px; resize: vertical; }
        form button {
            background-color: #4a90d9;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            font-weight: bold;
        }
        form button:hover { background-color: #357abd; }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            color: #4a90d9;
            text-decoration: none;
        }
        .back-link:hover { text-decoration: underline; }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        .empty-state h2 { margin-bottom: 10px; }
        .full-note { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .full-note h2 { color: #4a90d9; margin-bottom: 15px; font-size: 1.8em; }
        .full-note .content { white-space: pre-wrap; line-height: 1.8; }
        .full-note .meta { color: #999; font-size: 0.9em; margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee; }
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
        {% block body %}{% endblock %}
    </div>
</body>
</html>
'''

INDEX_TEMPLATE = '''
{% extends base %}
{% block body %}
{% if notes %}
    {% for note in notes %}
    <div class="note-card">
        <h2>{{ note['title'] }}</h2>
        <div class="content">{{ note['content'][:200] }}{% if note['content']|length > 200 %}...{% endif %}</div>
        <div class="meta">
            <span>Created: {{ note['created_at'] }} | Updated: {{ note['updated_at'] }}</span>
            <div class="actions">
                <a href="{{ url_for('view_note', note_id=note['id']) }}" class="btn-view">View</a>
                <a href="{{ url_for('edit_note', note_id=note['id']) }}" class="btn-edit">Edit</a>
                <a href="{{ url_for('delete_note', note_id=note['id']) }}" class="btn-delete" onclick="return confirm('Are you sure you want to delete this note?')">Delete</a>
            </div>
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
{% extends base %}
{% block body %}
<a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
<form method="POST">
    <h2 style="margin-bottom: 20px; color: #333;">Create New Note</h2>
    <label for="title">Title</label>
    <input type="text" id="title" name="title" placeholder="Enter note title..." required>
    <label for="content">Content</label>
    <textarea id="content" name="content" placeholder="Write your note here..." required></textarea>
    <button type="submit">Save Note</button>
</form>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends base %}
{% block body %}
<a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
<form method="POST">
    <h2 style="margin-bottom: 20px; color: #333;">Edit Note</h2>
    <label for="title">Title</label>
    <input type="text" id="title" name="title" value="{{ note['title'] }}" required>
    <label for="content">Content</label>
    <textarea id="content" name="content" required>{{ note['content'] }}</textarea>
    <button type="submit">Update Note</button>
</form>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends base %}
{% block body %}
<a href="{{ url_for('index') }}" class="back-link">← Back to Notes</a>
<div class="full-note">
    <h2>{{ note['title'] }}</h2>
    <div class="content">{{ note['content'] }}</div>
    <div class="meta">
        Created: {{ note['created_at'] }} | Last updated: {{ note['updated_at'] }}
    </div>
</div>
{% endblock %}
'''


@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes ORDER BY updated_at DESC').fetchall()
    conn.close()
    return render_template_string(INDEX_TEMPLATE, notes=notes, base=BASE_TEMPLATE)


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
    return render_template_string(CREATE_TEMPLATE, base=BASE_TEMPLATE)


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
            return redirect(url_for('index'))
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template_string(EDIT_TEMPLATE, note=note, base=BASE_TEMPLATE)


@app.route('/view/<int:note_id>')
def view_note(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if note is None:
        return redirect(url_for('index'))
    return render_template_string(VIEW_TEMPLATE, note=note, base=BASE_TEMPLATE)


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