from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

# Simple in-memory storage for notes
notes = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Notes</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
        .note-card { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px; }
        input[type="text"], textarea { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 4px; }
        textarea { height: 150px; }
        .btn { background: #007bff; color: white; padding: 10px 15px; text-decoration: none; border-radius: 3px; border: none; cursor: pointer; }
        .btn-edit { background: #28a745; }
        .nav { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="{{ url_for('index') }}">Home</a> | 
        <a href="{{ url_for('create') }}">Create New Note</a>
    </div>

    {% if page == 'index' %}
        <h1>My Notes</h1>
        {% for note in notes %}
            <div class="note-card">
                <h3>{{ note.title }}</h3>
                <p>{{ note.content[:100] }}{{ '...' if note.content|length > 100 else '' }}</p>
                <a href="{{ url_for('view', note_id=loop.index0) }}">View</a> | 
                <a href="{{ url_for('edit', note_id=loop.index0) }}">Edit</a>
            </div>
        {% endfor %}
    {% elif page == 'form' %}
        <h1>{{ 'Edit' if note_id is not none else 'Create' }} Note</h1>
        <form method="POST">
            <input type="text" name="title" placeholder="Note Title" value="{{ note.title if note else '' }}" required>
            <textarea name="content" placeholder="Write your note here..." required>{{ note.content if note else '' }}</textarea>
            <button type="submit" class="btn">Save Note</button>
        </form>
    {% elif page == 'view' %}
        <h1>{{ note.title }}</h1>
        <div style="white-space: pre-wrap;">{{ note.content }}</div>
        <br>
        <a href="{{ url_for('edit', note_id=note_id) }}" class="btn btn-edit">Edit Note</a>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, page='index', notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index'))
    return render_template_string(HTML_TEMPLATE, page='form', note=None, note_id=None)

@app.route('/view/<int:note_id>')
def view(note_id):
    if 0 <= note_id < len(notes):
        return render_template_string(HTML_TEMPLATE, page='view', note=notes[note_id], note_id=note_id)
    return redirect(url_for('index'))

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if 0 <= note_id < len(notes):
        if request.method == 'POST':
            notes[note_id]['title'] = request.form.get('title')
            notes[note_id]['content'] = request.form.get('content')
            return redirect(url_for('view', note_id=note_id))
        return render_template_string(HTML_TEMPLATE, page='form', note=notes[note_id], note_id=note_id)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)