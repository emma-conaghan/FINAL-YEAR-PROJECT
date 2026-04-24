from flask import Flask, request, render_template_string, redirect, url_for
from jinja2 import DictLoader

app = Flask(__name__)

# In-memory storage for notes
notes = []

# HTML Templates
TEMPLATES = {
    "base": """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Notes App</title>
        <style>
            body { font-family: sans-serif; margin: 40px; line-height: 1.6; max-width: 800px; margin: auto; background: #f4f4f4; }
            .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .note-item { border-bottom: 1px solid #eee; padding: 10px 0; }
            .note-item:last-child { border-bottom: none; }
            form div { margin-bottom: 15px; }
            input[type="text"], textarea { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
            button { background: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .nav { margin-bottom: 20px; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav">
                <a href="{{ url_for('index') }}">Home</a> | 
                <a href="{{ url_for('create_note') }}">New Note</a>
            </div>
            {% block content %}{% endblock %}
        </div>
    </body>
    </html>
    """,
    "index.html": """
    {% extends "base" %}
    {% block content %}
        <h1>My Notes</h1>
        {% if not notes %}
            <p>No notes found. Create one!</p>
        {% endif %}
        {% for note in notes %}
            <div class="note-item">
                <h3><a href="{{ url_for('view_note', note_id=loop.index0) }}">{{ note.title }}</a></h3>
                <p>{{ note.content[:100] }}{{ '...' if note.content|length > 100 else '' }}</p>
                <small><a href="{{ url_for('edit_note', note_id=loop.index0) }}">Edit</a></small>
            </div>
        {% endfor %}
    {% endblock %}
    """,
    "form.html": """
    {% extends "base" %}
    {% block content %}
        <h1>{{ action }} Note</h1>
        <form method="POST">
            <div>
                <label>Title</label>
                <input type="text" name="title" value="{{ note.title if note else '' }}" required>
            </div>
            <div>
                <label>Content</label>
                <textarea name="content" rows="10" required>{{ note.content if note else '' }}</textarea>
            </div>
            <button type="submit">Save Note</button>
        </form>
    {% endblock %}
    """,
    "view.html": """
    {% extends "base" %}
    {% block content %}
        <h1>{{ note.title }}</h1>
        <div style="white-space: pre-wrap; background: #fff; padding: 15px; border: 1px solid #ddd;">{{ note.content }}</div>
        <br>
        <a href="{{ url_for('edit_note', note_id=note_id) }}">Edit this note</a>
    {% endblock %}
    """
}

app.jinja_loader = DictLoader(TEMPLATES)

@app.route('/')
def index():
    return render_template_string(TEMPLATES["index.html"], notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index'))
    return render_template_string(TEMPLATES["form.html"], action="Create", note=None)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    if note_id >= len(notes):
        return "Note not found", 404
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title')
        notes[note_id]['content'] = request.form.get('content')
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(TEMPLATES["form.html"], action="Edit", note=notes[note_id])

@app.route('/view/<int:note_id>')
def view_note(note_id):
    if note_id >= len(notes):
        return "Note not found", 404
    return render_template_string(TEMPLATES["view.html"], note=notes[note_id], note_id=note_id)

if __name__ == '__main__':
    app.run(debug=True)