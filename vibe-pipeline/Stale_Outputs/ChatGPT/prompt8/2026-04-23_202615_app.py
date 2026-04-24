from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

base_template = """
<!doctype html>
<title>Notes App</title>
<h1><a href="{{ url_for('index') }}">Notes App</a></h1>
{% block body %}{% endblock %}
"""

index_template = """
{% extends "base" %}
{% block body %}
<p><a href="{{ url_for('create') }}">Create a New Note</a></p>
<ul>
  {% for id, note in notes.items() %}
    <li><a href="{{ url_for('view', note_id=id) }}">{{ note['title'] or '(No Title)' }}</a> 
    - <a href="{{ url_for('edit', note_id=id) }}">Edit</a></li>
  {% else %}
    <li>No notes yet.</li>
  {% endfor %}
</ul>
{% endblock %}
"""

form_template = """
{% extends "base" %}
{% block body %}
<form method="post">
  <p><label>Title:<br><input type="text" name="title" value="{{ note.title if note else '' }}"></label></p>
  <p><label>Content:<br><textarea name="content" rows="10" cols="40">{{ note.content if note else '' }}</textarea></label></p>
  <p><button type="submit">Save</button></p>
</form>
<p><a href="{{ url_for('index') }}">Back to notes</a></p>
{% endblock %}
"""

view_template = """
{% extends "base" %}
{% block body %}
<h2>{{ note.title or '(No Title)' }}</h2>
<pre>{{ note.content or '' }}</pre>
<p><a href="{{ url_for('edit', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to notes</a></p>
{% endblock %}
"""

@app.route("/")
def index():
    return render_template_string(index_template, notes=notes)

@app.route("/create", methods=["GET", "POST"])
def create():
    global next_id
    if request.method == "POST":
        title = request.form.get("title", "")
        content = request.form.get("content", "")
        notes[next_id] = {"title": title, "content": content}
        next_id += 1
        return redirect(url_for("index"))
    return render_template_string(form_template, note=None)

@app.route("/edit/<int:note_id>", methods=["GET", "POST"])
def edit(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    if request.method == "POST":
        note["title"] = request.form.get("title", "")
        note["content"] = request.form.get("content", "")
        return redirect(url_for("view", note_id=note_id))
    return render_template_string(form_template, note=type("NoteObj", (object,), note)())

@app.route("/view/<int:note_id>")
def view(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    return render_template_string(view_template, note=type("NoteObj", (object,), note)(), note_id=note_id)

@app.context_processor
def inject_base():
    return dict(base=base_template)

if __name__ == "__main__":
    app.run(debug=True)