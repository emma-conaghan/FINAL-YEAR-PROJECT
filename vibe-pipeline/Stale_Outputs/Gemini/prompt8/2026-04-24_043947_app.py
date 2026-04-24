from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

notes = []

BASE_LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Notes</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .note { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        form div { margin-bottom: 15px; }
        textarea { width: 100%; height: 100px; }
        input[type="text"] { width: 100%; }
    </style>
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>
"""

INDEX_TEMPLATE = """
{% extends "base" %}
{% block content %}
    <h1>My Notes</h1>
    <a href="{{ url_for('create_note') }}">Add New Note</a>
    <hr>
    {% if not notes %}
        <p>No notes yet.</p>
    {% endif %}
    {% for note in notes %}
        <div class="note">
            <h3>{{ note.title }}</h3>
            <p>{{ note.content }}</p>
            <a href="{{ url_for('edit_note', index=loop.index0) }}">Edit</a>
        </div>
    {% endfor %}
{% endblock %}
"""

FORM_TEMPLATE = """
{% extends "base" %}
{% block content %}
    <h1>{{ 'Edit' if edit_mode else 'Create' }} Note</h1>
    <form method="POST">
        <div>
            <label>Title</label><br>
            <input type="text" name="title" value="{{ note.title if note else '' }}" required>
        </div>
        <div>
            <label>Content</label><br>
            <textarea name="content" required>{{ note.content if note else '' }}</textarea>
        </div>
        <button type="submit">Save Note</button>
        <a href="{{ url_for('index') }}">Cancel</a>
    </form>
{% endblock %}
"""

@app.route('/_base')
def base():
    return render_template_string(BASE_LAYOUT)

@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE, notes=notes, base=BASE_LAYOUT)

@app.route('/create', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index'))
    return render_template_string(FORM_TEMPLATE, edit_mode=False, note=None)

@app.route('/edit/<int:index>', methods=['GET', 'POST'])
def edit_note(index):
    if index < 0 or index >= len(notes):
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        notes[index]['title'] = request.form.get('title')
        notes[index]['content'] = request.form.get('content')
        return redirect(url_for('index'))
    
    return render_template_string(FORM_TEMPLATE, edit_mode=True, note=notes[index])

# Helper to mock template inheritance in a single file
@app.context_processor
def inject_base():
    return {'base': 'base_layout.html'}

def render_template_string_with_base(template_str, **context):
    # This manually combines templates since we aren't using a file system
    full_template = template_str.replace('{% extends "base" %}', BASE_LAYOUT)
    return render_template_string(full_template, **context)

# Overriding original index/create/edit for the mock inheritance
@app.route('/')
def index_fixed():
    return render_template_string_with_base(INDEX_TEMPLATE, notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create_note_fixed():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index_fixed'))
    return render_template_string_with_base(FORM_TEMPLATE, edit_mode=False, note=None)

@app.route('/edit/<int:index>', methods=['GET', 'POST'])
def edit_note_fixed(index):
    if index < 0 or index >= len(notes):
        return redirect(url_for('index_fixed'))
    if request.method == 'POST':
        notes[index]['title'] = request.form.get('title')
        notes[index]['content'] = request.form.get('content')
        return redirect(url_for('index_fixed'))
    return render_template_string_with_base(FORM_TEMPLATE, edit_mode=True, note=notes[index])

if __name__ == '__main__':
    app.run(debug=True)