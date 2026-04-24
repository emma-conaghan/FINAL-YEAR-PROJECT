from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<ul>
{% for nid, note in notes.items() %}
  <li><a href="{{ url_for('view_note', note_id=nid) }}">{{ note['title'] }}</a></li>
{% else %}
  <li>No notes yet</li>
{% endfor %}
</ul>
<a href="{{ url_for('create_note') }}">Create a new note</a>
'''

VIEW_HTML = '''
<!doctype html>
<title>{{ note['title'] }}</title>
<h1>{{ note['title'] }}</h1>
<p>{{ note['content']|replace('\n', '<br>')|safe }}</p>
<a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> |
<a href="{{ url_for('index') }}">Back to all notes</a>
'''

FORM_HTML = '''
<!doctype html>
<title>{{ "Edit" if note else "Create" }} Note</title>
<h1>{{ "Edit" if note else "Create" }} Note</h1>
<form method="post">
  <label for="title">Title</label><br>
  <input id="title" name="title" value="{{ note['title'] if note else '' }}" required><br><br>
  <label for="content">Content</label><br>
  <textarea id="content" name="content" rows="10" cols="30" required>{{ note['content'] if note else '' }}</textarea><br><br>
  <button type="submit">Save</button>
</form>
<a href="{{ url_for('index') }}">Back to all notes</a>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/note/create', methods=['GET', 'POST'])
def create_note():
    global next_id
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            notes[next_id] = {'title': title, 'content': content}
            next_id += 1
            return redirect(url_for('index'))
    return render_template_string(FORM_HTML, note=None)

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            notes[note_id] = {'title': title, 'content': content}
            return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(FORM_HTML, note=note)

if __name__ == '__main__':
    app.run(debug=True)