from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)
notes = []

LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Notes App</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; line-height: 1.6; padding: 0 20px; }
        nav { margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .note-item { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
        input[type="text"], textarea { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button { background: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 4px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        <a href="{{ url_for('create') }}">Create Note</a>
    </nav>
    <main>{{ content | safe }}</main>
</body>
</html>
"""

INDEX_CONTENT = """
<h1>Notes</h1>
{% if not notes %}
    <p>No notes found. Create one!</p>
{% else %}
    {% for i in range(notes|length) %}
        <div class="note-item">
            <h3>{{ notes[i].title }}</h3>
            <a href="{{ url_for('view', note_id=i) }}">View</a> | 
            <a href="{{ url_for('edit', note_id=i) }}">Edit</a>
        </div>
    {% endfor %}
{% endif %}
"""

VIEW_CONTENT = """
<h1>{{ note.title }}</h1>
<pre>{{ note.content }}</pre>
<hr>
<a href="{{ url_for('edit', note_id=note_id) }}">Edit Note</a>
"""

FORM_CONTENT = """
<h1>{{ action }} Note</h1>
<form method="post">
    <label>Title</label>
    <input type="text" name="title" value="{{ note.title if note else '' }}" required>
    <label>Content</label>
    <textarea name="content" rows="10" required>{{ note.content if note else '' }}</textarea>
    <button type="submit">Save Note</button>
</form>
"""

@app.route('/')
def index():
    body = render_template_string(INDEX_CONTENT, notes=notes)
    return render_template_string(LAYOUT, content=body)

@app.route('/note/<int:note_id>')
def view(note_id):
    if 0 <= note_id < len(notes):
        body = render_template_string(VIEW_CONTENT, note=notes[note_id], note_id=note_id)
        return render_template_string(LAYOUT, content=body)
    return redirect(url_for('index'))

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index'))
    body = render_template_string(FORM_CONTENT, action="Create", note=None)
    return render_template_string(LAYOUT, content=body)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if 0 <= note_id < len(notes):
        if request.method == 'POST':
            notes[note_id]['title'] = request.form.get('title')
            notes[note_id]['content'] = request.form.get('content')
            return redirect(url_for('view', note_id=note_id))
        body = render_template_string(FORM_CONTENT, action="Edit", note=notes[note_id])
        return render_template_string(LAYOUT, content=body)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)