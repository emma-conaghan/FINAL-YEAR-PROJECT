from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

list_template = '''
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('create_note') }}">Create New Note</a>
<ul>
{% for id, note in notes.items() %}
  <li><a href="{{ url_for('view_note', note_id=id) }}">{{ note['title'] }}</a> - <a href="{{ url_for('edit_note', note_id=id) }}">Edit</a></li>
{% else %}
  <li>No notes yet.</li>
{% endfor %}
</ul>
'''

view_template = '''
<!doctype html>
<title>{{ note['title'] }}</title>
<h1>{{ note['title'] }}</h1>
<p>{{ note['content'] | nl2br }}</p>
<a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to Notes</a>
'''

form_template = '''
<!doctype html>
<title>{{ title }}</title>
<h1>{{ title }}</h1>
<form method="post">
  <label for="title">Title:</label><br>
  <input id="title" name="title" required value="{{ note.get('title','') }}"><br><br>
  <label for="content">Content:</label><br>
  <textarea id="content" name="content" rows="10" cols="30" required>{{ note.get('content','') }}</textarea><br><br>
  <button type="submit">Save</button>
</form>
<a href="{{ url_for('index') }}">Back to Notes</a>
'''

@app.template_filter('nl2br')
def nl2br_filter(s):
    return s.replace('\n', '<br>\n')

@app.route('/')
def index():
    return render_template_string(list_template, notes=notes)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    return render_template_string(view_template, note=note, note_id=note_id)

@app.route('/note/new', methods=['GET', 'POST'])
def create_note():
    global next_id
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title:
            notes[next_id] = {'title': title, 'content': content}
            next_id += 1
            return redirect(url_for('index'))
    return render_template_string(form_template, title="Create Note", note={})

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title:
            note['title'] = title
            note['content'] = content
            return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(form_template, title="Edit Note", note=note)

if __name__ == '__main__':
    app.run(debug=True)