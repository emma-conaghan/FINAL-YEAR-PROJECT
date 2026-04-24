from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)
notes = {}
next_id = 1

INDEX_HTML = '''
<!DOCTYPE html>
<html>
<head><title>Notes</title></head>
<body>
<h1>Notes</h1>
<a href="{{ url_for('create') }}">Create New Note</a>
<ul>
{% for nid, note in notes.items() %}
  <li><a href="{{ url_for('view', note_id=nid) }}">{{ note['title'] }}</a></li>
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
<head><title>{{ note['title'] }}</title></head>
<body>
<h1>{{ note['title'] }}</h1>
<p>{{ note['content']|replace('\n','<br>')|safe }}</p>
<a href="{{ url_for('edit', note_id=note_id) }}">Edit</a> | <a href="{{ url_for('index') }}">Back to list</a>
</body>
</html>
'''

FORM_HTML = '''
<!DOCTYPE html>
<html>
<head><title>{{ 'Edit' if note_id else 'Create' }} Note</title></head>
<body>
<h1>{{ 'Edit' if note_id else 'Create' }} Note</h1>
<form method="post">
  <label>Title:<br><input type="text" name="title" value="{{ note['title'] if note else '' }}" required></label><br><br>
  <label>Content:<br><textarea name="content" rows="10" cols="30" required>{{ note['content'] if note else '' }}</textarea></label><br><br>
  <input type="submit" value="Save">
</form>
<a href="{{ url_for('index') }}">Back to list</a>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes)

@app.route('/note/<int:note_id>')
def view(note_id):
    note = notes.get(note_id)
    if not note:
        return redirect(url_for('index'))
    return render_template_string(VIEW_HTML, note=note, note_id=note_id)

@app.route('/create', methods=['GET', 'POST'])
def create():
    global next_id
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            notes[next_id] = {'title': title, 'content': content}
            next_id += 1
            return redirect(url_for('index'))
    return render_template_string(FORM_HTML, note=None, note_id=None)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    note = notes.get(note_id)
    if not note:
        return redirect(url_for('index'))
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        if title and content:
            notes[note_id] = {'title': title, 'content': content}
            return redirect(url_for('view', note_id=note_id))
    return render_template_string(FORM_HTML, note=note, note_id=note_id)

if __name__ == '__main__':
    app.run(debug=True)