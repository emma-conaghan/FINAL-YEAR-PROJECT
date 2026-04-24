from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)
notes = {}
note_id_counter = 1

template_index = '''
<!DOCTYPE html>
<html>
<head>
    <title>Personal Notes</title>
</head>
<body>
    <h1>Personal Notes</h1>
    <a href="{{ url_for('add_note') }}">Create Note</a>
    <ul>
    {% for nid, note in notes.items() %}
        <li>
            <a href="{{ url_for('view_note', note_id=nid) }}">{{ note['title'] }}</a>
            (<a href="{{ url_for('edit_note', note_id=nid) }}">Edit</a>)
        </li>
    {% endfor %}
    </ul>
</body>
</html>
'''

template_view = '''
<!DOCTYPE html>
<html>
<head>
    <title>View Note</title>
</head>
<body>
    <h2>{{ note['title'] }}</h2>
    <p>{{ note['content'] }}</p>
    <a href="{{ url_for('index') }}">Back</a>
</body>
</html>
'''

template_form = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ form_title }}</title>
</head>
<body>
    <h2>{{ form_title }}</h2>
    <form method="post">
        <label>Title:<br>
            <input type="text" name="title" value="{{ note_title }}" required>
        </label><br><br>
        <label>Content:<br>
            <textarea name="content" rows="8" cols="40" required>{{ note_content }}</textarea>
        </label><br><br>
        <button type="submit">Save</button>
    </form>
    <a href="{{ url_for('index') }}">Back</a>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(template_index, notes=notes)

@app.route('/note/<int:note_id>')
def view_note(note_id):
    note = notes.get(note_id)
    if note:
        return render_template_string(template_view, note=note)
    return 'Note not found', 404

@app.route('/add', methods=['GET', 'POST'])
def add_note():
    global note_id_counter
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        notes[note_id_counter] = {'title': title, 'content': content}
        note_id_counter += 1
        return redirect(url_for('index'))
    return render_template_string(template_form, form_title='Create Note', note_title='', note_content='')

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    note = notes.get(note_id)
    if not note:
        return 'Note not found', 404
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        notes[note_id] = {'title': title, 'content': content}
        return redirect(url_for('index'))
    return render_template_string(template_form, form_title='Edit Note', note_title=note['title'], note_content=note['content'])

if __name__ == '__main__':
    app.run(debug=True)