from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)
notes = {}
note_id_counter = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>All Notes</h1>
<a href="{{ url_for('create_note') }}">Create a new note</a>
<ul>
  {% for nid, note in notes.items() %}
    <li><a href="{{ url_for('view_note', note_id=nid) }}">{{ note['title'] or "Untitled" }}</a> - <a href="{{ url_for('edit_note', note_id=nid) }}">Edit</a></li>
  {% else %}
    <li><em>No notes yet</em></li>
  {% endfor %}
</ul>
'''

CREATE_EDIT_HTML = '''
<!doctype html>
<title>{% if note %}Edit{% else %}Create{% endif %} Note</title>
<h1>{% if note %}Edit{% else %}Create{% endif %} Note</h1>
<form method="post">
  <p><input type="text" name="title" placeholder="Title" value="{{ note.title if note else '' }}" style="width: 300px;"></p>
  <p><textarea name="content" placeholder="Content" rows="10" cols="40">{{ note.content if note else '' }}</textarea></p>
  <p><button type="submit">Save</button></p>
</form>
<a href="{{ url_for('index') }}">Back to all notes</a>
'''

VIEW_HTML = '''
<!doctype html>
<title>{{ note.title or "Untitled" }}</title>
<h1>{{ note.title or "Untitled" }}</h1>
<div style="white-space: pre-wrap;">{{ note.content }}</div>
<p><a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to all notes</a></p>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/create', methods=['GET', 'POST'])
def create_note():
    global note_id_counter
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        notes[note_id_counter] = {'title': title, 'content': content}
        note_id_counter += 1
        return redirect(url_for('index'))
    return render_template_string(CREATE_EDIT_HTML, note=None)

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        note['title'] = title
        note['content'] = content
        return redirect(url_for('view_note', note_id=note_id))
    class NoteObj:
        def __init__(self, d):
            self.title = d['title']
            self.content = d['content']
    return render_template_string(CREATE_EDIT_HTML, note=NoteObj(note))

@app.route('/note/<int:note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

if __name__ == '__main__':
    app.run(debug=True)