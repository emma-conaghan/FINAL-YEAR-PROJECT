from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

# Global list to store notes as dictionaries
notes = []

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Personal Notes</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; line-height: 1.6; }
        .note { border: 1px solid #eee; padding: 15px; margin-bottom: 20px; border-radius: 5px; box-shadow: 2px 2px 5px #f9f9f9; }
        .note h3 { margin-top: 0; }
        form input, form textarea { width: 100%; padding: 10px; margin-bottom: 10px; box-sizing: border-box; }
        button { padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; }
        .nav { margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
    </style>
</head>
<body>
    <div class="nav">
        <strong>NotesApp</strong> | 
        <a href="{{ url_for('index') }}">View All</a> | 
        <a href="{{ url_for('create') }}">Add Note</a>
    </div>
    {% if content %}
        {{ content|safe }}
    {% endif %}
</body>
</html>
"""

INDEX_CONTENT = """
<h1>Your Notes</h1>
{% for note in notes %}
<div class="note">
    <h3>{{ note.title }}</h3>
    <p>{{ note.content }}</p>
    <a href="{{ url_for('edit', note_id=loop.index0) }}">Edit Note</a>
</div>
{% else %}
<p>No notes found. Start by creating one!</p>
{% endfor %}
"""

FORM_CONTENT = """
<h1>{{ 'Edit' if note else 'Create' }} Note</h1>
<form method="POST">
    <input name="title" placeholder="Note Title" value="{{ note.title if note else '' }}" required>
    <textarea name="content" placeholder="Write your note here..." rows="8" required>{{ note.content if note else '' }}</textarea>
    <button type="submit">Save Note</button>
</form>
"""

@app.route('/')
def index():
    content = render_template_string(INDEX_CONTENT, notes=notes)
    return render_template_string(BASE_TEMPLATE, content=content)

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index'))
    
    html_form = render_template_string(FORM_CONTENT, note=None)
    return render_template_string(BASE_TEMPLATE, content=html_form)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if note_id < 0 or note_id >= len(notes):
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title')
        notes[note_id]['content'] = request.form.get('content')
        return redirect(url_for('index'))
    
    html_form = render_template_string(FORM_CONTENT, note=notes[note_id])
    return render_template_string(BASE_TEMPLATE, content=html_form)

if __name__ == '__main__':
    app.run(debug=True)