from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)
notes = {}
next_id = 1

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Notes</title>
</head>
<body>
    <h1>Personal Notes</h1>
    <a href="{{ url_for('new_note') }}">Create New Note</a>
    <ul>
    {% for note_id, note in notes.items() %}
        <li><a href="{{ url_for('view_note', note_id=note_id) }}">{{ note['title'] or 'Untitled' }}</a> - 
        <a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a></li>
    {% else %}
        <li>No notes yet.</li>
    {% endfor %}
    </ul>
</body>
</html>
"""

NEW_EDIT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ "Edit" if note else "New" }} Note</title>
</head>
<body>
    <h1>{{ "Edit" if note else "New" }} Note</h1>
    <form method="post">
        <label>Title:<br><input type="text" name="title" value="{{ note['title'] if note else '' }}" style="width:300px"></label><br><br>
        <label>Content:<br><textarea name="content" rows="10" cols="50">{{ note['content'] if note else '' }}</textarea></label><br><br>
        <button type="submit">Save</button>
    </form>
    <a href="{{ url_for('index') }}">Back to Notes</a>
</body>
</html>
"""

VIEW_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ note['title'] or 'Untitled' }}</title>
</head>
<body>
    <h1>{{ note['title'] or 'Untitled' }}</h1>
    <pre style="white-space: pre-wrap;">{{ note['content'] }}</pre>
    <a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to Notes</a>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route("/note/new", methods=["GET", "POST"])
def new_note():
    global next_id
    if request.method == "POST":
        title = request.form.get("title", "")
        content = request.form.get("content", "")
        notes[next_id] = {"title": title, "content": content}
        next_id += 1
        return redirect(url_for("index"))
    return render_template_string(NEW_EDIT_HTML, note=None)

@app.route("/note/<int:note_id>")
def view_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found.", 404
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route("/note/<int:note_id>/edit", methods=["GET", "POST"])
def edit_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found.", 404
    if request.method == "POST":
        title = request.form.get("title", "")
        content = request.form.get("content", "")
        note["title"] = title
        note["content"] = content
        return redirect(url_for("view_note", note_id=note_id))
    return render_template_string(NEW_EDIT_HTML, note=note)

if __name__ == "__main__":
    app.run(debug=True)