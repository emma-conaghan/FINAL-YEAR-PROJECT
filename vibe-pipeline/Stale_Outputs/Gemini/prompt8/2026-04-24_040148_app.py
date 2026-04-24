import sqlite3
from flask import Flask, request, redirect, render_template_string

app = Flask(__name__)
DATABASE = 'notes.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT)')

@app.route('/')
def index():
    conn = get_db()
    notes = conn.execute('SELECT * FROM notes').fetchall()
    conn.close()
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Notes</title></head>
    <body>
        <h1>My Notes</h1>
        <a href="/create">Create New Note</a>
        <hr>
        <ul>
            {% for note in notes %}
            <li>
                <strong><a href="/view/{{ note['id'] }}">{{ note['title'] }}</a></strong>
                [<a href="/edit/{{ note['id'] }}">Edit</a>]
            </li>
            {% endfor %}
        </ul>
    </body>
    </html>
    """
    return render_template_string(html, notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title', 'Untitled')
        content = request.form.get('content', '')
        conn = get_db()
        conn.execute('INSERT INTO notes (title, content) VALUES (?, ?)', (title, content))
        conn.commit()
        conn.close()
        return redirect('/')
    html = """
    <h1>Create Note</h1>
    <form method="post">
        <p>Title:<br><input type="text" name="title" style="width:100%"></p>
        <p>Content:<br><textarea name="content" rows="10" style="width:100%"></textarea></p>
        <p><button type="submit">Save</button> <a href="/">Cancel</a></p>
    </form>
    """
    return render_template_string(html)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    conn = get_db()
    if request.method == 'POST':
        title = request.form.get('title', 'Untitled')
        content = request.form.get('content', '')
        conn.execute('UPDATE notes SET title = ?, content = ? WHERE id = ?', (title, content, note_id))
        conn.commit()
        conn.close()
        return redirect('/')
    
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if not note:
        return "Note not found", 404

    html = """
    <h1>Edit Note</h1>
    <form method="post">
        <p>Title:<br><input type="text" name="title" value="{{ note['title'] }}" style="width:100%"></p>
        <p>Content:<br><textarea name="content" rows="10" style="width:100%">{{ note['content'] }}</textarea></p>
        <p><button type="submit">Update</button> <a href="/">Cancel</a></p>
    </form>
    """
    return render_template_string(html, note=note)

@app.route('/view/<int:note_id>')
def view(note_id):
    conn = get_db()
    note = conn.execute('SELECT * FROM notes WHERE id = ?', (note_id,)).fetchone()
    conn.close()
    if not note:
        return "Note not found", 404

    html = """
    <h1>{{ note['title'] }}</h1>
    <div style="white-space: pre-wrap; border: 1px solid #ccc; padding: 10px;">{{ note['content'] }}</div>
    <p><a href="/edit/{{ note['id'] }}">Edit</a> | <a href="/">Back to list</a></p>
    """
    return render_template_string(html, note=note)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)