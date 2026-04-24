from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('create') }}">Create New Note</a>
<ul>
  {% for note_id, note in notes.items() %}
    <li><a href="{{ url_for('view', note_id=note_id) }}">{{ note['title'] }}</a></li>
  {% else %}
    <li><em>No notes yet.</em></li>
  {% endfor %}
</ul>
'''

VIEW_HTML = '''
<!doctype html>
<title>{{ note['title'] }}</title>
<h1>{{ note['title'] }}</h1>
<p>{{ note['content'].replace('\\n','<br>')|safe }}</p>
<a href="{{ url_for('edit', note_id=note_id) }}">Edit</a> |
<a href="{{ url_for('index') }}">Back to Notes</a>
'''

EDIT_HTML = '''
<!doctype html>
<title>{% if note %}Edit{% else %}Create{% endif %} Note</title>
<h1>{% if note %}Edit{% else %}Create{% endif %} Note</h1>
<form method="post">
  <label>Title:<br><input type="text" name="title" value="{{ note['title'] if note else '' }}"></label><br><br>
  <label>Content:<br><textarea name="content" rows="10" cols="30">{{ note['content'] if note else '' }}</textarea></label><br><br>
  <button type="submit">Save</button>
</form>
<a href="{{ url_for('index') }}">Back to Notes</a>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/note/new', methods=['GET', 'POST'])
def create():
    global next_id
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title == '':
            title = 'Untitled'
        notes[next_id] = {'title': title, 'content': content}
        note_id = next_id
        next_id += 1
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(EDIT_HTML, note=None)

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit(note_id):
    note = notes.get(note_id)
    if not note:
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title == '':
            title = 'Untitled'
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(EDIT_HTML, note=note)

if __name__ == '__main__':
    app.run(debug=True)