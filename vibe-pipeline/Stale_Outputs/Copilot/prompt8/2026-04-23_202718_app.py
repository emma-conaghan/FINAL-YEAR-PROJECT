from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

notes = {}
note_id = 1

note_list_html = """
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('new_note') }}">Create New Note</a>
<ul>
{% for n_id, note in notes.items() %}
    <li>
        <a href="{{ url_for('view_note', note_id=n_id) }}">{{ note['title'] }}</a>
        (<a href="{{ url_for('edit_note', note_id=n_id) }}">Edit</a>)
    </li>
{% endfor %}
</ul>
"""

note_form_html = """
<!doctype html>
<title>{{ form_title }}</title>
<h1>{{ form_title }}</h1>
<form method="post">
  Title: <input type="text" name="title" value="{{ note_title }}"><br>
  Content:<br>
  <textarea name="content" rows="8" cols="40">{{ note_content }}</textarea><br>
  <input type="submit" value="Save">
</form>
<a href="{{ url_for('list_notes') }}">Back to Notes</a>
"""

view_note_html = """
<!doctype html>
<title>{{ note['title'] }}</title>
<h1>{{ note['title'] }}</h1>
<pre>{{ note['content'] }}</pre>
<a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> |
<a href="{{ url_for('list_notes') }}">Back to Notes</a>
"""

@app.route("/")
def list_notes():
    return render_template_string(note_list_html, notes=notes)

@app.route("/note/new", methods=["GET", "POST"])
def new_note():
    global note_id
    if request.method == "POST":
        title = request.form.get("title", "")
        content = request.form.get("content", "")
        notes[note_id] = {"title": title, "content": content}
        nid = note_id
        note_id += 1
        return redirect(url_for('view_note', note_id=nid))
    return render_template_string(note_form_html, form_title="Create Note", note_title="", note_content="")

@app.route("/note/<int:note_id>", methods=["GET"])
def view_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found.", 404
    return render_template_string(view_note_html, note=note, note_id=note_id)

@app.route("/note/<int:note_id>/edit", methods=["GET", "POST"])
def edit_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found.", 404
    if request.method == "POST":
        title = request.form.get("title", "")
        content = request.form.get("content", "")
        notes[note_id] = {"title": title, "content": content}
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(note_form_html, form_title="Edit Note", note_title=note["title"], note_content=note["content"])

if __name__ == "__main__":
    app.run(debug=True)