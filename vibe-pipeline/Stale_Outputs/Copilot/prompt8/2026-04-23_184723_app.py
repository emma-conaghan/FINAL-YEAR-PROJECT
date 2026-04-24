from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = {}

template_index = '''
<!doctype html>
<title>Notes</title>
<h1>Personal Notes</h1>
<a href="{{ url_for('new_note') }}">Create New Note</a>
<ul>
{% for id, note in notes.items() %}
  <li>
    <a href="{{ url_for('view_note', note_id=id) }}">{{ note['title'] }}</a>
    (<a href="{{ url_for('edit_note', note_id=id) }}">edit</a>)
  </li>
{% else %}
  <li>No notes yet.</li>
{% endfor %}
</ul>
'''

template_note = '''
<!doctype html>
<title>{{ note['title'] }}</title>
<h1>{{ note['title'] }}</h1>
<div>{{ note['content'] }}</div>
<p><a href="{{ url_for('index') }}">Back to Notes</a></p>
'''

template_form = '''
<!doctype html>
<title>{% if edit %}Edit{% else %}New{% endif %} Note</title>
<h1>{% if edit %}Edit{% else %}New{% endif %} Note</h1>
<form method="post">
  <p>Title:<br><input type="text" name="title" value="{{ note.get('title','') }}">
  <p>Content:<br><textarea name="content" rows="6" cols="40">{{ note.get('content','') }}</textarea>
  <p><input type="submit" value="{% if edit %}Save{% else %}Create{% endif %}">
</form>
<p><a href="{{ url_for('index') }}">Back to Notes</a></p>
'''

@app.route('/')
def index():
    return render_template_string(template_index, notes=notes)

@app.route('/note/new', methods=['GET', 'POST'])
def new_note():
    if request.method == 'POST':
        title = request.form.get('title','').strip()
        content = request.form.get('content','').strip()
        note_id = str(len(notes) + 1)
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(template_form, note={}, edit=False)

@app.route('/note/<note_id>', methods=['GET'])
def view_note(note_id):
    note = notes.get(note_id)
    if not note:
        return 'Note not found.', 404
    return render_template_string(template_note, note=note)

@app.route('/note/<note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if not note:
        return 'Note not found.', 404
    if request.method == 'POST':
        title = request.form.get('title','').strip()
        content = request.form.get('content','').strip()
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('view_note', note_id=note_id))
    return render_template_string(template_form, note=note, edit=True)

if __name__ == '__main__':
    app.run(debug=True)