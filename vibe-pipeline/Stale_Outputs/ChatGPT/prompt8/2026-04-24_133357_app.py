from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

INDEX_HTML = """
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('create') }}">Create New Note</a>
<ul>
{% for note_id, note in notes.items() %}
  <li><a href="{{ url_for('view', note_id=note_id) }}">{{ note['title'] }}</a></li>
{% else %}
  <li>No notes yet.</li>
{% endfor %}
</ul>
"""

CREATE_HTML = """
<!doctype html>
<title>Create Note</title>
<h1>Create Note</h1>
<form method="post">
  <p>Title:<br><input type="text" name="title" required></p>
  <p>Content:<br><textarea name="content" rows="10" cols="40" required></textarea></p>
  <p><button type="submit">Save</button> <a href="{{ url_for('index') }}">Cancel</a></p>
</form>
"""

VIEW_HTML = """
<!doctype html>
<title>{{ note['title'] }}</title>
<h1>{{ note['title'] }}</h1>
<p>{{ note['content'] | nl2br }}</p>
<p><a href="{{ url_for('edit', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to Notes</a></p>
"""

EDIT_HTML = """
<!doctype html>
<title>Edit Note</title>
<h1>Edit Note</h1>
<form method="post">
  <p>Title:<br><input type="text" name="title" value="{{ note['title'] }}" required></p>
  <p>Content:<br><textarea name="content" rows="10" cols="40" required>{{ note['content'] }}</textarea></p>
  <p><button type="submit">Save</button> <a href="{{ url_for('view', note_id=note_id) }}">Cancel</a></p>
</form>
"""

@app.template_filter('nl2br')
def nl2br_filter(s):
    return s.replace('\n', '<br>\n')

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create():
    global next_id
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        notes[next_id] = {'title': title, 'content': content}
        next_id += 1
        return redirect(url_for('index'))
    return render_template_string(CREATE_HTML)

@app.route('/note/<int:note_id>')
def view(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(EDIT_HTML, note=note, note_id=note_id)

if __name__ == '__main__':
    app.run(debug=True)