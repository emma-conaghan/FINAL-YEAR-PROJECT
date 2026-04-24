from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('create_note') }}">Create New Note</a>
<ul>
{% for id, note in notes.items() %}
  <li><a href="{{ url_for('view_note', note_id=id) }}">{{ note['title'] or 'Untitled' }}</a></li>
{% else %}
  <li>No notes yet</li>
{% endfor %}
</ul>
'''

VIEW_HTML = '''
<!doctype html>
<title>{{ note['title'] or 'Untitled' }}</title>
<h1>{{ note['title'] or 'Untitled' }}</h1>
<p>{{ note['content']|nl2br }}</p>
<a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> |
<a href="{{ url_for('index') }}">Back to list</a>
'''

FORM_HTML = '''
<!doctype html>
<title>{{ mode }} Note</title>
<h1>{{ mode }} Note</h1>
<form method="post">
  <p><input type="text" name="title" placeholder="Title" value="{{ note.get('title', '') }}">
  <p><textarea name="content" rows="10" cols="30" placeholder="Content">{{ note.get('content', '') }}</textarea>
  <p><button type="submit">Save</button>
</form>
<a href="{{ url_for('index') }}">Back to list</a>
'''

@app.template_filter('nl2br')
def nl2br_filter(s):
    return s.replace('\n', '<br>')

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/create', methods=['GET', 'POST'])
def create_note():
    global next_id
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        notes[next_id] = {'title': title, 'content': content}
        next_id += 1
        return redirect(url_for('index'))
    return render_template_string(FORM_HTML, note={}, mode="Create")

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(FORM_HTML, note=note, mode="Edit")

if __name__ == '__main__':
    app.run(debug=True)