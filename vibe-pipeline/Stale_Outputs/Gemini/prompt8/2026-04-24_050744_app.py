import sqlite3
from flask import Flask, request, redirect, render_template_string

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('notes.db')
    conn.execute('CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT)')
    conn.close()

def query_db(query, args=(), one=False):
    conn = sqlite3.connect('notes.db')
    conn.row_factory = sqlite3.Row
    cur = conn.execute(query, args)
    rv = cur.fetchall()
    conn.commit()
    conn.close()
    return (rv[0] if rv else None) if one else rv

@app.route('/')
def index():
    notes = query_db('SELECT * FROM notes')
    html = """
    <h1>My Notes</h1>
    <a href="/new">Create New Note</a>
    <ul>
        {% for note in notes %}
        <li>
            <strong>{{ note.title }}</strong> 
            [<a href="/edit/{{ note.id }}">Edit</a>]
            <p>{{ note.content }}</p>
        </li>
        {% endfor %}
    </ul>
    """
    return render_template_string(html, notes=notes)

@app.route('/new', methods=['GET', 'POST'])
def new_note():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        query_db('INSERT INTO notes (title, content) VALUES (?, ?)', (title, content))
        return redirect('/')
    
    html = """
    <h1>New Note</h1>
    <form method="post">
        Title: <input type="text" name="title"><br>
        Content:<br><textarea name="content"></textarea><br>
        <button type="submit">Save</button>
    </form>
    <a href="/">Back</a>
    """
    return render_template_string(html)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        query_db('UPDATE notes SET title = ?, content = ? WHERE id = ?', (title, content, note_id))
        return redirect('/')
    
    note = query_db('SELECT * FROM notes WHERE id = ?', (note_id,), one=True)
    html = """
    <h1>Edit Note</h1>
    <form method="post">
        Title: <input type="text" name="title" value="{{ note.title }}"><br>
        Content:<br><textarea name="content">{{ note.content }}</textarea><br>
        <button type="submit">Update</button>
    </form>
    <a href="/">Back</a>
    """
    return render_template_string(html, note=note)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)