from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)
notes = {}
next_id = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('create') }}">Create a new note</a>
<ul>
{% for note_id, note in notes.items() %}
  <li><a href="{{ url_for('view', note_id=note_id) }}">{{ note['title'] or 'Untitled' }}</a></li>
{% else %}
  <li>No notes yet</li>
{% endfor %}
</ul>
'''

VIEW_HTML = '''
<!doctype html>
<title>{{ note['title'] or 'Untitled' }}</title>
<h1>{{ note['title'] or 'Untitled' }}</h1>
<p>{{ note['content'] | nl2br }}</p>
<a href="{{ url_for('edit', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to list</a>
'''

EDIT_HTML = '''
<!doctype html>
<title>{{ 'Edit' if note_id is not none else 'Create' }} Note</title>
<h1>{{ 'Edit' if note_id is not none else 'Create' }} Note</h1>
<form method="post">
  <p><input type="text" name="title" placeholder="Title" value="{{ note['title'] or '' }}"></p>
  <p><textarea name="content" placeholder="Content" rows="10" cols="50">{{ note['content'] or '' }}</textarea></p>
  <p><button type="submit">Save</button></p>
</form>
<a href="{{ url_for('index') }}">Back to list</a>
'''

@app.template_filter('nl2br')
def nl2br_filter(s):
    return s.replace('\n', '<br>\n')

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view(note_id):
    note = notes.get(note_id)
    if not note:
        return redirect(url_for('index'))
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit(note_id):
    note = notes.get(note_id)
    if not note:
        return redirect(url_for('index'))
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title', '').strip()
        notes[note_id]['content'] = request.form.get('content', '')
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(EDIT_HTML, note=note, note_id=note_id)

@app.route('/create', methods=['GET', 'POST'])
def create():
    global next_id
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '')
        notes[next_id] = {'title': title, 'content': content}
        new_id = next_id
        next_id += 1
        return redirect(url_for('view', note_id=new_id))
    empty_note = {'title': '', 'content': ''}
    return render_template_string(EDIT_HTML, note=empty_note, note_id=None)

if __name__ == '__main__':
    app.run(debug=True)