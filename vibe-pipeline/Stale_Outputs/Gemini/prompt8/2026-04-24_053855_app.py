from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

# Simple in-memory storage for notes
notes = []

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Personal Notes</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
        .note { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
        .note h3 { margin-top: 0; }
        form input, form textarea { width: 100%; padding: 10px; margin-bottom: 10px; box-sizing: border-box; }
        .actions { margin-top: 10px; }
        a { color: #007bff; text-decoration: none; }
        .btn { background: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
    </style>
</head>
<body>
    <h1>Personal Notes</h1>
    {% block content %}{% endblock %}
</body>
</html>
"""

INDEX_TEMPLATE = """
{% extends "base" %}
{% block content %}
    <a href="{{ url_for('create_note') }}" class="btn">Create New Note</a>
    <hr>
    {% if not notes %}
        <p>No notes found. Create one!</p>
    {% else %}
        {% for note in notes %}
            <div class="note">
                <h3>{{ note.title }}</h3>
                <p>{{ note.content }}</p>
                <div class="actions">
                    <a href="{{ url_for('edit_note', note_id=loop.index0) }}">Edit</a>
                </div>
            </div>
        {% endfor %}
    {% endif %}
{% endblock %}
"""

FORM_TEMPLATE = """
{% extends "base" %}
{% block content %}
    <h2>{{ action }} Note</h2>
    <form method="POST">
        <input type="text" name="title" placeholder="Note Title" value="{{ note.title if note else '' }}" required>
        <textarea name="content" rows="10" placeholder="Note Content..." required>{{ note.content if note else '' }}</textarea>
        <button type="submit" class="btn">Save Note</button>
        <a href="{{ url_for('index') }}" style="margin-left: 10px;">Cancel</a>
    </form>
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE, notes=notes, base=BASE_TEMPLATE)

@app.route('/create', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index'))
    return render_template_string(FORM_TEMPLATE, action="Create", note=None, base=BASE_TEMPLATE)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    if note_id < 0 or note_id >= len(notes):
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title')
        notes[note_id]['content'] = request.form.get('content')
        return redirect(url_for('index'))
    
    return render_template_string(FORM_TEMPLATE, action="Edit", note=notes[note_id], base=BASE_TEMPLATE)

# Helper for template inheritance simulation in render_template_string
@app.context_processor
def inject_base():
    return {'base': BASE_TEMPLATE}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)