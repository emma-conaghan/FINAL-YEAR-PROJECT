from flask import Flask, request, redirect, render_template_string

app = Flask(__name__)

# Simple in-memory storage for notes
notes = []

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Personal Notes</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .note { border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        form div { margin-bottom: 10px; }
        textarea { width: 100%; height: 100px; }
        input[type="text"] { width: 100%; }
    </style>
</head>
<body>
    <h1><a href="/" style="text-decoration:none; color:black;">My Notes</a></h1>
    <a href="/new">Create New Note</a>
    <hr>
    {{ content | safe }}
</body>
</html>
"""

@app.route('/')
def index():
    if not notes:
        content = "<p>No notes found.</p>"
    else:
        content = ""
        for i, note in enumerate(notes):
            content += f"""
            <div class="note">
                <h3>{note['title']}</h3>
                <p>{note['content']}</p>
                <a href="/edit/{i}">Edit</a>
            </div>
            """
    return render_template_string(HTML_LAYOUT, content=content)

@app.route('/new', methods=['GET', 'POST'])
def new_note():
    if request.method == 'POST':
        title = request.form.get('title', 'Untitled')
        body = request.form.get('content', '')
        notes.append({'title': title, 'content': body})
        return redirect('/')
    
    form = """
    <form method="POST">
        <div>Title:<br><input type="text" name="title" required></div>
        <div>Content:<br><textarea name="content" required></textarea></div>
        <button type="submit">Save Note</button>
    </form>
    """
    return render_template_string(HTML_LAYOUT, content=form)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    if note_id >= len(notes):
        return redirect('/')
        
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title', 'Untitled')
        notes[note_id]['content'] = request.form.get('content', '')
        return redirect('/')
    
    note = notes[note_id]
    form = f"""
    <form method="POST">
        <div>Title:<br><input type="text" name="title" value="{note['title']}" required></div>
        <div>Content:<br><textarea name="content" required>{note['content']}</textarea></div>
        <button type="submit">Update Note</button>
        <a href="/">Cancel</a>
    </form>
    """
    return render_template_string(HTML_LAYOUT, content=form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)