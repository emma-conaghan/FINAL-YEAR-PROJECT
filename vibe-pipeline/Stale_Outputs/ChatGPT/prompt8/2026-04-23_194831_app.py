from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
note_id_counter = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>All Notes</h1>
<a href="{{ url_for('create') }}">Create New Note</a>
<ul>
  {% for nid, note in notes.items() %}
    <li><a href="{{ url_for('view', note_id=nid) }}">{{ note['title'] or 'Untitled' }}</a> - <a href="{{ url_for('edit', note_id=nid) }}">Edit</a></li>
  {% endfor %}
</ul>
'''

VIEW_HTML = '''
<!doctype html>
<title>{{ note['title'] or 'Untitled' }}</title>
<h1>{{ note['title'] or 'Untitled' }}</h1>
<p>{{ note['content'] or '(No content)' }}</p>
<a href="{{ url_for('edit', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to all notes</a>
'''

FORM_HTML = '''
<!doctype html>
<title>{{ form_title }}</title>
<h1>{{ form_title }}</h1>
<form method="post">
  <label for="title">Title:</label><br>
  <input type="text" id="title" name="title" value="{{ note.get('title', '') }}"><br><br>
  <label for="content">Content:</label><br>
  <textarea id="content" name="content" rows="10" cols="30">{{ note.get('content', '') }}</textarea><br><br>
  <input type="submit" value="Save">
</form>
<a href="{{ url_for('index') }}">Back to all notes</a>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found.", 404
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/note/create', methods=['GET', 'POST'])
def create():
    global note_id_counter
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        notes[note_id_counter] = {'title': title, 'content': content}
        note_id_counter += 1
        return redirect(url_for('index'))
    return render_template_string(FORM_HTML, form_title="Create Note", note={})

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found.", 404
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        note['title'] = title
        note['content'] = content
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(FORM_HTML, form_title="Edit Note", note=note)

if __name__ == '__main__':
    app.run(debug=True)