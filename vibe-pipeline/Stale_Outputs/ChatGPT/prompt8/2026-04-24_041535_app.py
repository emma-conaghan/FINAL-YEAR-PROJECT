from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
note_id_counter = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('create_note') }}">Create New Note</a>
<ul>
  {% for nid, note in notes.items() %}
    <li><a href="{{ url_for('view_note', note_id=nid) }}">{{ note['title'] or 'Untitled' }}</a></li>
  {% else %}
    <li>No notes yet.</li>
  {% endfor %}
</ul>
'''

VIEW_HTML = '''
<!doctype html>
<title>{{ note['title'] or 'Untitled' }}</title>
<h1>{{ note['title'] or 'Untitled' }}</h1>
<p>{{ note['content'] or '' }}</p>
<a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a>
<a href="{{ url_for('index') }}">Back to list</a>
'''

EDIT_HTML = '''
<!doctype html>
<title>{{ 'Edit Note' if note else 'Create Note' }}</title>
<h1>{{ 'Edit Note' if note else 'Create Note' }}</h1>
<form method="post">
  <label>Title: <input type="text" name="title" value="{{ note['title'] if note else '' }}"></label><br><br>
  <label>Content:<br>
    <textarea name="content" rows="10" cols="40">{{ note['content'] if note else '' }}</textarea>
  </label><br><br>
  <input type="submit" value="Save">
</form>
<a href="{{ url_for('index') }}">Back to list</a>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(EDIT_HTML, note=note)

@app.route('/note/create', methods=['GET', 'POST'])
def create_note():
    global note_id_counter
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        notes[note_id_counter] = {'title': title, 'content': content}
        note_id_counter += 1
        return redirect(url_for('index'))
    return render_template_string(EDIT_HTML, note=None)

if __name__ == '__main__':
    app.run(debug=True)