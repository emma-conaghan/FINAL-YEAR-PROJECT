from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)

# Simple in-memory database
notes = {}
note_id_counter = 1

# Base HTML layout used across the app
LAYOUT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Personal Notes</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #333; }
        .note-card { border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 8px; background: #fafafa; }
        .note-card h3 { margin-top: 0; }
        input[type="text"], textarea { width: 100%; padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 16px; }
        textarea { height: 150px; }
        button { background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        .nav { margin-bottom: 20px; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="nav">
        <strong><a href="/">My Notes</a></strong>
    </div>
    {{ content|safe }}
</body>
</html>
"""

@app.route('/')
def index():
    note_list_html = ""
    for nid, note in notes.items():
        note_list_html += f"""
        <div class="note-card">
            <h3>{note['title']}</h3>
            <a href="/view/{nid}">View</a> | <a href="/edit/{nid}">Edit</a>
        </div>
        """
    
    main_content = f"""
    <h2>Create Note</h2>
    <form action="/create" method="POST">
        <input type="text" name="title" placeholder="Note Title" required>
        <textarea name="text" placeholder="Start writing..."></textarea>
        <button type="submit">Save Note</button>
    </form>
    <hr>
    <h2>Saved Notes</h2>
    {note_list_html if note_list_html else "<p>No notes yet. Create one above!</p>"}
    """
    return render_template_string(LAYOUT_TEMPLATE, content=main_content)

@app.route('/create', methods=['POST'])
def create():
    global note_id_counter
    title = request.form.get('title')
    text = request.form.get('text')
    notes[note_id_counter] = {'title': title, 'text': text}
    note_id_counter += 1
    return redirect(url_for('index'))

@app.route('/view/<int:note_id>')
def view(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    
    view_content = f"""
    <h2>{note['title']}</h2>
    <p style="white-space: pre-wrap;">{note['text']}</p>
    <br>
    <a href="/">&laquo; Back to List</a>
    """
    return render_template_string(LAYOUT_TEMPLATE, content=view_content)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    
    if request.method == 'POST':
        note['title'] = request.form.get('title')
        note['text'] = request.form.get('text')
        return redirect(url_for('index'))
    
    edit_form = render_template_string("""
    <h2>Edit Note</h2>
    <form method="POST">
        <input type="text" name="title" value="{{ note.title }}" required>
        <textarea name="text">{{ note.text }}</textarea>
        <button type="submit">Update Note</button>
    </form>
    <br>
    <a href="/">Cancel</a>
    """, note=note)
    
    return render_template_string(LAYOUT_TEMPLATE, content=edit_form)

if __name__ == '__main__':
    app.run(debug=True)