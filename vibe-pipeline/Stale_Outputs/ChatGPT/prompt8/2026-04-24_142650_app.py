from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
note_id_counter = 1

index_html = '''
<!doctype html>
<title>Notes</title>
<h1>My Notes</h1>
<a href="{{ url_for('create') }}">Create New Note</a>
<ul>
  {% for nid, note in notes.items() %}
    <li>
      <a href="{{ url_for('view', note_id=nid) }}">{{ note['title'] or "Untitled" }}</a> -
      <a href="{{ url_for('edit', note_id=nid) }}">Edit</a>
    </li>
  {% else %}
    <li>No notes yet.</li>
  {% endfor %}
</ul>
'''

form_html = '''
<!doctype html>
<title>{{ title }}</title>
<h1>{{ title }}</h1>
<form method="post">
  <label for="title">Title</label><br>
  <input type="text" id="title" name="title" value="{{ note.title|default('') }}"><br><br>
  <label for="content">Content</label><br>
  <textarea id="content" name="content" rows="10" cols="30">{{ note.content|default('') }}</textarea><br><br>
  <input type="submit" value="Save">
</form>
<a href="{{ url_for('index') }}">Back to notes</a>
'''

view_html = '''
<!doctype html>
<title>{{ note.title or "Untitled" }}</title>
<h1>{{ note.title or "Untitled" }}</h1>
<p>{{ note.content|nl2br }}</p>
<a href="{{ url_for('index') }}">Back to notes</a> |
<a href="{{ url_for('edit', note_id=note_id) }}">Edit</a>
'''

@app.template_filter('nl2br')
def nl2br_filter(s):
    return s.replace('\n','<br>')

@app.route('/')
def index():
    return render_template_string(index_html, notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create():
    global note_id_counter
    if request.method == 'POST':
        title = request.form.get('title','')
        content = request.form.get('content','')
        notes[note_id_counter] = {'title': title, 'content': content}
        note_id_counter += 1
        return redirect(url_for('index'))
    return render_template_string(form_html, title="Create Note", note={})

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if note_id not in notes:
        return redirect(url_for('index'))
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title','')
        notes[note_id]['content'] = request.form.get('content','')
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(form_html, title="Edit Note", note=notes[note_id])

@app.route('/view/<int:note_id>')
def view(note_id):
    note = notes.get(note_id)
    if not note:
        return redirect(url_for('index'))
    return render_template_string(view_html, note=note, note_id=note_id)

if __name__ == '__main__':
    app.run(debug=True)