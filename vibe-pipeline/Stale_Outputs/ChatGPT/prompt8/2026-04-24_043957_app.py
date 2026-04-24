from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}
note_id_counter = 1

INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head><title>Notes</title></head>
<body>
<h1>Notes</h1>
<a href="{{ url_for('create') }}">Create New Note</a>
<ul>
{% for id, note in notes.items() %}
  <li><a href="{{ url_for('view', note_id=id) }}">{{ note['title'] or 'Untitled' }}</a> -
  <a href="{{ url_for('edit', note_id=id) }}">Edit</a></li>
{% else %}
  <li>No notes yet.</li>
{% endfor %}
</ul>
</body>
</html>
'''

VIEW_HTML = '''
<!DOCTYPE html>
<html>
<head><title>{{ note.title or 'Untitled' }}</title></head>
<body>
<h1>{{ note.title or 'Untitled' }}</h1>
<p>{{ note.content or '' }}</p>
<p><a href="{{ url_for('index') }}">Back</a> | <a href="{{ url_for('edit', note_id=note_id) }}">Edit</a></p>
</body>
</html>
'''

FORM_HTML = '''
<!DOCTYPE html>
<html>
<head><title>{{ 'Edit Note' if edit else 'New Note' }}</title></head>
<body>
<h1>{{ 'Edit Note' if edit else 'New Note' }}</h1>
<form method="post">
  <label>Title: <input type="text" name="title" value="{{ note.title }}"></label><br><br>
  <label>Content:<br>
  <textarea name="content" rows="10" cols="50">{{ note.content }}</textarea></label><br><br>
  <input type="submit" value="Save">
</form>
<p><a href="{{ url_for('index') }}">Back</a></p>
</body>
</html>
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
    global note_id_counter
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        notes[note_id_counter] = {'title': title, 'content': content}
        note_id_counter += 1
        return redirect(url_for('index'))
    empty_note = {'title': '', 'content': ''}
    return render_template_string(FORM_HTML, note=empty_note, edit=False)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    note = notes.get(note_id)
    if note is None:
        return redirect(url_for('index'))
    if request.method == 'POST':
        note['title'] = request.form.get('title', '').strip()
        note['content'] = request.form.get('content', '').strip()
        return redirect(url_for('view', note_id=note_id))
    return render_template_string(FORM_HTML, note=note, edit=True)

if __name__ == '__main__':
    app.run(debug=True)