from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
note_id_counter = 1

INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head><title>Notes</title></head>
<body>
<h1>Personal Notes</h1>
<a href="{{ url_for('create_note') }}">Create New Note</a>
<ul>
{% for nid, note in notes.items() %}
  <li><a href="{{ url_for('view_note', note_id=nid) }}">{{ note['title'] }}</a> - <a href="{{ url_for('edit_note', note_id=nid) }}">Edit</a></li>
{% else %}
  <li>No notes yet.</li>
{% endfor %}
</ul>
</body>
</html>
'''

NOTE_HTML = '''
<!DOCTYPE html>
<html>
<head><title>{{ note['title'] }}</title></head>
<body>
<h1>{{ note['title'] }}</h1>
<p>{{ note['content']|replace('\n','<br>')|safe }}</p>
<a href="{{ url_for('index') }}">Back to notes</a> | 
<a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a>
</body>
</html>
'''

FORM_HTML = '''
<!DOCTYPE html>
<html>
<head><title>{{ form_title }}</title></head>
<body>
<h1>{{ form_title }}</h1>
<form method="post">
  <label for="title">Title:</label><br>
  <input type="text" name="title" id="title" value="{{ note.title if note else '' }}" required><br><br>
  <label for="content">Content:</label><br>
  <textarea name="content" id="content" rows="10" cols="30" required>{{ note.content if note else '' }}</textarea><br><br>
  <button type="submit">Save</button>
</form>
<a href="{{ url_for('index') }}">Back to notes</a>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    return render_template_string(NOTE_HTML, note=note, note_id=note_id)

@app.route('/note/create', methods=['GET', 'POST'])
def create_note():
    global note_id_counter
    if request.method == 'POST':
        title = request.form.get('title','').strip()
        content = request.form.get('content','').strip()
        if title and content:
            notes[note_id_counter] = {'title': title, 'content': content}
            note_id_counter += 1
            return redirect(url_for('index'))
        else:
            return "Title and content required", 400
    return render_template_string(FORM_HTML, form_title="Create Note", note=None)

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if note is None:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title','').strip()
        content = request.form.get('content','').strip()
        if title and content:
            notes[note_id] = {'title': title, 'content': content}
            return redirect(url_for('view_note', note_id=note_id))
        else:
            return "Title and content required", 400
    # For GET, pass note as an object to support attribute access in template
    class NoteObj:
        def __init__(self, d):
            self.title = d['title']
            self.content = d['content']
    return render_template_string(FORM_HTML, form_title="Edit Note", note=NoteObj(note))

if __name__ == '__main__':
    app.run(debug=True)