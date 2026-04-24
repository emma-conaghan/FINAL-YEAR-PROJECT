from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)

notes = []

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Personal Notes</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .note { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        form { margin-bottom: 20px; }
        input, textarea { width: 100%; margin-bottom: 10px; display: block; padding: 8px; }
        textarea { height: 100px; }
        .btn { background: #007BFF; color: white; border: none; padding: 10px 15px; cursor: pointer; text-decoration: none; display: inline-block; }
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
    <h3>Create New Note</h3>
    <form action="{{ url_for('create') }}" method="POST">
        <input type="text" name="title" placeholder="Title" required>
        <textarea name="content" placeholder="Note content..." required></textarea>
        <button type="submit" class="btn">Save Note</button>
    </form>
    <hr>
    <h3>Your Saved Notes</h3>
    {% for id, note in notes.items() %}
        <div class="note">
            <h4>{{ note.title }}</h4>
            <p>{{ note.content }}</p>
            <a href="{{ url_for('edit', note_id=id) }}">Edit</a>
        </div>
    {% endfor %}
{% endblock %}
"""

EDIT_TEMPLATE = """
{% extends "base" %}
{% block content %}
    <h3>Edit Note</h3>
    <form method="POST">
        <input type="text" name="title" value="{{ note.title }}" required>
        <textarea name="content" required>{{ note.content }}</textarea>
        <button type="submit" class="btn">Update Note</button>
        <a href="{{ url_for('index') }}">Cancel</a>
    </form>
{% endblock %}
"""

storage = {}
counter = 0

@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE, notes=storage), 200, {'Content-Type': 'text/html'}

@app.route('/create', methods=['POST'])
def create():
    global counter
    title = request.form.get('title')
    content = request.form.get('content')
    storage[counter] = {'title': title, 'content': content}
    counter += 1
    return redirect(url_for('index'))

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if note_id not in storage:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        storage[note_id]['title'] = request.form.get('title')
        storage[note_id]['content'] = request.form.get('content')
        return redirect(url_for('index'))
    
    return render_template_string(EDIT_TEMPLATE, note=storage[note_id]), 200, {'Content-Type': 'text/html'}

@app.context_processor
def inject_templates():
    return dict(base=BASE_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)