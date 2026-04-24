from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>All Notes</h1>
<a href="{{ url_for('create') }}">Create New Note</a>
<ul>
  {% for note_id, note in notes.items() %}
  <li><a href="{{ url_for('view', note_id=note_id) }}">{{ note['title'] or 'Untitled' }}</a></li>
  {% else %}
  <li>No notes yet.</li>
  {% endfor %}
</ul>
'''

VIEW_HTML = '''
<!doctype html>
<title>{{ note['title'] or 'Untitled' }}</title>
<h1>{{ note['title'] or 'Untitled' }}</h1>
<p>{{ note['content']|replace('\n', '<br>')|safe }}</p>
<p><a href="{{ url_for('index') }}">Back to all notes</a> | <a href="{{ url_for('edit', note_id=note_id) }}">Edit Note</a></p>
'''

FORM_HTML = '''
<!doctype html>
<title>{% if note_id %}Edit{% else %}Create{% endif %} Note</title>
<h1>{% if note_id %}Edit{% else %}Create{% endif %} Note</h1>
<form method="post">
  <p><input type="text" name="title" placeholder="Title" value="{{ note['title'] if note else '' }}" style="width:300px"></p>
  <p><textarea name="content" placeholder="Content" rows="10" cols="40">{{ note['content'] if note else '' }}</textarea></p>
  <p><button type="submit">Save</button> <a href="{{ url_for('index') }}">Cancel</a></p>
</form>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view(note_id):
    note = notes.get(note_id)
    if note is None:
        return redirect(url_for('index'))
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/create', methods=['GET', 'POST'])
def create():
    global next_id
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        notes[next_id] = {'title': title, 'content': content}
        next_id += 1
        return redirect(url_for('index'))
    return render_template_string(FORM_HTML, note=None, note_id=None)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    note = notes.get(note_id)
    if note is None:
        return redirect(url_for('index'))
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(FORM_HTML, note=note, note_id=note_id)

if __name__ == '__main__':
    app.run(debug=True)