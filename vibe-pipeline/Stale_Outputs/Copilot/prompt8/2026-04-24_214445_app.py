from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = []

template_index = '''
<!doctype html>
<title>Notes</title>
<h1>Your Notes</h1>
<a href="{{ url_for('create') }}">Create New Note</a>
<ul>
{% for idx, note in enumerate(notes) %}
  <li>
    <a href="{{ url_for('view', note_id=idx) }}">{{ note['title'] }}</a>
    (<a href="{{ url_for('edit', note_id=idx) }}">edit</a>)
  </li>
{% endfor %}
</ul>
'''

template_create = '''
<!doctype html>
<title>Create Note</title>
<h1>Create Note</h1>
<form method="post">
  Title: <input type="text" name="title"><br>
  Content:<br>
  <textarea name="content" rows="10" cols="40"></textarea><br>
  <input type="submit" value="Save">
</form>
<a href="{{ url_for('index') }}">Back</a>
'''

template_edit = '''
<!doctype html>
<title>Edit Note</title>
<h1>Edit Note</h1>
<form method="post">
  Title: <input type="text" name="title" value="{{ note['title'] }}"><br>
  Content:<br>
  <textarea name="content" rows="10" cols="40">{{ note['content'] }}</textarea><br>
  <input type="submit" value="Save">
</form>
<a href="{{ url_for('index') }}">Back</a>
'''

template_view = '''
<!doctype html>
<title>View Note</title>
<h1>{{ note['title'] }}</h1>
<pre>{{ note['content'] }}</pre>
<a href="{{ url_for('index') }}">Back</a>
<a href="{{ url_for('edit', note_id=note_id) }}">Edit</a>
'''

@app.route('/')
def index():
    return render_template_string(template_index, notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        notes.append({'title': title, 'content': content})
        return redirect(url_for('index'))
    return render_template_string(template_create)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if note_id < 0 or note_id >= len(notes):
        return "Note not found", 404
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(template_edit, note=notes[note_id])

@app.route('/view/<int:note_id>')
def view(note_id):
    if note_id < 0 or note_id >= len(notes):
        return "Note not found", 404
    return render_template_string(template_view, note=notes[note_id], note_id=note_id)

if __name__ == '__main__':
    app.run(debug=True)