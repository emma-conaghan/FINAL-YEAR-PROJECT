from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

# In-memory storage for notes
notes = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Notes</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; line-height: 1.5; }
        .note { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px; }
        .note h3 { margin-top: 0; }
        form div { margin-bottom: 15px; }
        input[type="text"], textarea { width: 100%; padding: 8px; box-sizing: border-box; }
        textarea { height: 120px; }
        .nav { margin-bottom: 20px; }
        button { background: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 4px; cursor: pointer; }
        .cancel { color: #666; margin-left: 10px; text-decoration: none; }
    </style>
</head>
<body>
    <h1>{{ page_title }}</h1>
    
    {% if view == 'list' %}
        <div class="nav">
            <a href="{{ url_for('create') }}">+ Create New Note</a>
        </div>
        {% for note in notes %}
            <div class="note">
                <h3>{{ note.title }}</h3>
                <p>{{ note.content }}</p>
                <a href="{{ url_for('edit', note_id=loop.index0) }}">Edit Note</a>
            </div>
        {% else %}
            <p>No notes found. Create one!</p>
        {% endfor %}
    {% else %}
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
            <a class="cancel" href="{{ url_for('index') }}">Cancel</a>
        </form>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE, 
        page_title="My Personal Notes", 
        notes=notes, 
        view='list'
    )

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index'))
    return render_template_string(
        HTML_TEMPLATE, 
        page_title="Create Note", 
        view='form', 
        note=None
    )

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if note_id < 0 or note_id >= len(notes):
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title')
        notes[note_id]['content'] = request.form.get('content')
        return redirect(url_for('index'))
        
    return render_template_string(
        HTML_TEMPLATE, 
        page_title="Edit Note", 
        view='form', 
        note=notes[note_id]
    )

if __name__ == '__main__':
    app.run(debug=True)