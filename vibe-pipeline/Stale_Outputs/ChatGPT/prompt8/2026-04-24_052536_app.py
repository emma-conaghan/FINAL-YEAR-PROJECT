from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}

INDEX_HTML = """
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('create_note') }}">Create New Note</a>
<ul>
{% for note_id, note in notes.items() %}
  <li><a href="{{ url_for('view_note', note_id=note_id) }}">{{ note['title'] or 'Untitled' }}</a></li>
{% else %}
  <li>No notes yet.</li>
{% endfor %}
</ul>
"""

VIEW_NOTE_HTML = """
<!doctype html>
<title>{{ note['title'] or 'Untitled' }}</title>
<h1>{{ note['title'] or 'Untitled' }}</h1>
<p>{{ note['content']|replace('\n','<br>')|safe }}</p>
<a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to all notes</a>
"""

EDIT_NOTE_HTML = """
<!doctype html>
<title>{{ 'Edit Note' if note else 'Create Note' }}</title>
<h1>{{ 'Edit Note' if note else 'Create Note' }}</h1>
<form method="post">
  <p><input type=text name=title placeholder="Title" value="{{ note['title'] if note else '' }}"></p>
  <p><textarea name=content placeholder="Content" rows=10 cols=30>{{ note['content'] if note else '' }}</textarea></p>
  <p><input type=submit value="{{ 'Update' if note else 'Create' }}"></p>
</form>
<a href="{{ url_for('index') }}">Back to all notes</a>
"""

def generate_id():
    return str(len(notes) + 1)

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    return render_template_string(VIEW_NOTE_HTML, note=note, note_id=note_id)

@app.route('/note/create', methods=['GET', 'POST'])
def create_note():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        note_id = generate_id()
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(EDIT_NOTE_HTML, note=None)

@app.route('/note/<note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        note['title'] = title
        note['content'] = content
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(EDIT_NOTE_HTML, note=note)

if __name__ == '__main__':
    app.run(debug=True)