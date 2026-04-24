from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
note_id_counter = 1

note_form = '''
<form method="post">
  Title:<br>
  <input type="text" name="title" value="{{ title|default('') }}"><br>
  Content:<br>
  <textarea name="content">{{ content|default('') }}</textarea><br>
  <input type="submit" value="Save">
</form>
'''

@app.route("/")
def index():
    notes_list = ""
    for note_id, note in notes.items():
        notes_list += f'<li><a href="/view/{note_id}">{note["title"]}</a> (<a href="/edit/{note_id}">Edit</a>)</li>'
    return render_template_string("""
    <h1>Notes</h1>
    <ul>
    {{ notes_list|safe }}
    </ul>
    <a href="/create">Create new note</a>
    """, notes_list=notes_list)

@app.route("/create", methods=["GET", "POST"])
def create():
    global note_id_counter
    if request.method == "POST":
        title = request.form.get("title", "")
        content = request.form.get("content", "")
        notes[note_id_counter] = {"title": title, "content": content}
        new_id = note_id_counter
        note_id_counter += 1
        return redirect(url_for("view_note", note_id=new_id))
    return render_template_string("<h2>Create Note</h2>" + note_form)

@app.route("/edit/<int:note_id>", methods=["GET", "POST"])
def edit(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    if request.method == "POST":
        title = request.form.get("title", "")
        content = request.form.get("content", "")
        notes[note_id] = {"title": title, "content": content}
        return redirect(url_for("view_note", note_id=note_id))
    return render_template_string("<h2>Edit Note</h2>" + note_form, title=note["title"], content=note["content"])

@app.route("/view/<int:note_id>")
def view_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    return render_template_string("""
    <h2>{{ title }}</h2>
    <pre>{{ content }}</pre>
    <a href="/">Back to notes</a> | <a href="/edit/{{ note_id }}">Edit</a>
    """, title=note["title"], content=note["content"], note_id=note_id)

if __name__ == "__main__":
    app.run(debug=True)