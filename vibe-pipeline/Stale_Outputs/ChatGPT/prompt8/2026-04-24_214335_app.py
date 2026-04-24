from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

INDEX_HTML = """
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('new_note') }}">Create New Note</a>
<ul>
  {% for nid, note in notes.items() %}
    <li><a href="{{ url_for('view_note', note_id=nid) }}">{{ note['title'] or 'Untitled' }}</a> - <a href="{{ url_for('edit_note', note_id=nid) }}">edit</a></li>
  {% else %}
    <li>No notes yet</li>
  {% endfor %}
</ul>
"""

VIEW_HTML = """
<!doctype html>
<title>{{ note.title or 'Untitled' }}</title>
<h1>{{ note.title or 'Untitled' }}</h1>
<p>{{ note.content.replace('\\n', '<br>')|safe }}</p>
<p><a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to list</a></p>
"""

EDIT_HTML = """
<!doctype html>
<title>{% if note %}Edit Note{% else %}New Note{% endif %}</title>
<h1>{% if note %}Edit Note{% else %}New Note{% endif %}</h1>
<form method="post">
  <p>Title:<br><input type="text" name="title" value="{{ note.title if note else '' }}"></p>
  <p>Content:<br><textarea name="content" rows="10" cols="30">{{ note.content if note else '' }}</textarea></p>
  <p><input type="submit" value="Save"></p>
</form>
<p><a href="{{ url_for('index') }}">Back to list</a></p>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route("/note/<int:note_id>")
def view_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return redirect(url_for("index"))
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route("/note/new", methods=["GET", "POST"])
def new_note():
    global next_id
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        notes[next_id] = {"title": title, "content": content}
        next_id += 1
        return redirect(url_for("index"))
    return render_template_string(EDIT_HTML, note=None)

@app.route("/note/<int:note_id>/edit", methods=["GET", "POST"])
def edit_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return redirect(url_for("index"))
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        notes[note_id] = {"title": title, "content": content}
        return redirect(url_for("view_note", note_id=note_id))
    return render_template_string(EDIT_HTML, note=note)

if __name__ == "__main__":
    app.run(debug=True)