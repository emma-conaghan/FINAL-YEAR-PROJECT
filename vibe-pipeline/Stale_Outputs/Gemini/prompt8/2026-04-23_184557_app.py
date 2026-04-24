from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

notes = []

LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Notes App</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #333; }
        .note { border: 1px solid #eee; padding: 15px; margin-bottom: 20px; border-radius: 8px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .note h3 { margin: 0 0 10px 0; color: #222; }
        .note p { margin: 0 0 10px 0; white-space: pre-wrap; }
        .form-group { margin-bottom: 15px; }
        label { display: block; font-weight: bold; margin-bottom: 5px; }
        input[type="text"], textarea { width: 100%; padding: 10px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; }
        button { background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #218838; }
        nav { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        a { color: #007bff; text-decoration: none; margin-right: 15px; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">View Notes</a>
        <a href="{{ url_for('create') }}">Create New Note</a>
    </nav>
    {{ content | safe }}
</body>
</html>
"""

INDEX_CONTENT = """
<h1>My Notes</h1>
{% for i in range(notes|length) %}
<div class="note">
    <h3>{{ notes[i].title }}</h3>
    <p>{{ notes[i].content }}</p>
    <a href="{{ url_for('edit', note_id=i) }}">Edit Note</a>
</div>
{% endfor %}
{% if not notes %}
<p>No notes found. Create your first one!</p>
{% endif %}
"""

FORM_CONTENT = """
<h1>{{ action }} Note</h1>
<form method="POST">
    <div class="form-group">
        <label>Title</label>
        <input type="text" name="title" value="{{ note.title if note else '' }}" required placeholder="Enter note title">
    </div>
    <div class="form-group">
        <label>Content</label>
        <textarea name="content" rows="8" required placeholder="Enter note text">{{ note.content if note else '' }}</textarea>
    </div>
    <button type="submit">Save Note</button>
</form>
"""

@app.route('/')
def index():
    body = render_template_string(INDEX_CONTENT, notes=notes)
    return render_template_string(LAYOUT, content=body)

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
    if note_id < 0 or note_id >= len(notes):
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title')
        notes[note_id]['content'] = request.form.get('content')
        return redirect(url_for('index'))
    
    body = render_template_string(FORM_CONTENT, action="Edit", note=notes[note_id])
    return render_template_string(LAYOUT, content=body)

if __name__ == '__main__':
    app.run(debug=True)