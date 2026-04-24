from flask import Flask, request, redirect, render_template_string

app = Flask(__name__)

# In-memory storage for notes
notes = []

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Notes</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
        .note-card { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 8px; }
        .note-card h3 { margin-top: 0; }
        textarea { width: 100%; height: 300px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        input[type="text"] { width: 100%; padding: 10px; margin-bottom: 20px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-size: 1.2em; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        nav { margin-bottom: 20px; }
        .meta { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a> | <a href="/create">Create New Note</a>
    </nav>
    <hr>
    {% block content %}{% endblock %}
</body>
</html>
"""

def render_page(content_html, **context):
    full_html = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', content_html)
    return render_template_string(full_html, **context)

@app.route('/')
def index():
    content = """
    <h1>My Notes</h1>
    {% for idx in range(notes|length) %}
    <div class="note-card">
        <h3><a href="/view/{{ idx }}">{{ notes[idx].title }}</a></h3>
        <div class="meta">
            <a href="/edit/{{ idx }}">Edit</a>
        </div>
    </div>
    {% endfor %}
    {% if not notes %}
    <p>You haven't created any notes yet.</p>
    {% endif %}
    """
    return render_page(content, notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title', 'Untitled')
        body = request.form.get('body', '')
        notes.append({'title': title, 'body': body})
        return redirect('/')
    
    content = """
    <h1>Create Note</h1>
    <form method="post">
        <input type="text" name="title" placeholder="Note Title" required>
        <textarea name="body" placeholder="Write your note here..." required></textarea>
        <p><button type="submit">Save Note</button></p>
    </form>
    """
    return render_page(content)

@app.route('/view/<int:note_id>')
def view(note_id):
    if 0 <= note_id < len(notes):
        note = notes[note_id]
        content = """
        <h1>{{ note.title }}</h1>
        <div style="white-space: pre-wrap; background: #f9f9f9; padding: 20px; border-radius: 8px;">{{ note.body }}</div>
        <p><a href="/edit/{{ note_id }}"><button>Edit Note</button></a></p>
        """
        return render_page(content, note=note, note_id=note_id)
    return "Note not found", 404

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if 0 <= note_id < len(notes):
        if request.method == 'POST':
            notes[note_id]['title'] = request.form.get('title', 'Untitled')
            notes[note_id]['body'] = request.form.get('body', '')
            return redirect(f'/view/{note_id}')
        
        note = notes[note_id]
        content = """
        <h1>Edit Note</h1>
        <form method="post">
            <input type="text" name="title" value="{{ note.title }}" required>
            <textarea name="body" required>{{ note.body }}</textarea>
            <p><button type="submit">Update Note</button></p>
        </form>
        """
        return render_page(content, note=note)
    return "Note not found", 404

if __name__ == '__main__':
    # Defaulting to debug mode for simplicity during development
    app.run(host='0.0.0.0', port=5000, debug=True)