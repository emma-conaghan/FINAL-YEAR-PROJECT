from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
next_id = 1

INDEX_HTML = '''
<!doctype html>
<title>Notes</title>
<h1>Notes</h1>
<a href="{{ url_for('new_note') }}">Create New Note</a>
<ul>
  {% for note_id, note in notes.items() %}
    <li><a href="{{ url_for('view_note', note_id=note_id) }}">{{ note['title'] or '(no title)' }}</a></li>
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
<a href="{{ url_for('edit_note', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to Notes</a>
'''

EDIT_HTML = '''
<!doctype html>
<title>{% if editing %}Edit Note{% else %}New Note{% endif %}</title>
<h1>{% if editing %}Edit Note{% else %}New Note{% endif %}</h1>
<form method="post">
  <label for="title">Title:</label><br>
  <input type="text" id="title" name="title" value="{{ note['title'] if note else '' }}"><br><br>
  <label for="content">Content:</label><br>
  <textarea id="content" name="content" rows="10" cols="30">{{ note['content'] if note else '' }}</textarea><br><br>
  <input type="submit" value="Save">
</form>
<a href="{{ url_for('index') }}">Back to Notes</a>
'''

@app.template_filter('nl2br')
def nl2br(value):
    return value.replace('\n', '<br>\n')

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if not note:
        return redirect(url_for('index'))
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/note/new', methods=['GET', 'POST'])
def new_note():
    global next_id
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        notes[next_id] = {'title': title, 'content': content}
        next_id += 1
        return redirect(url_for('index'))
    return render_template_string(EDIT_HTML, editing=False, note=None)

@app.route('/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if not note:
        return redirect(url_for('index'))
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        note['title'] = title
        note['content'] = content
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(EDIT_HTML, editing=True, note=note)

if __name__ == '__main__':
    app.run(debug=True)