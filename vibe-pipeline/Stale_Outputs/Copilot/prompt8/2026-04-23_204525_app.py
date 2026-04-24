from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

notes = []
note_id_counter = 1

layout = """
<!doctype html>
<title>Notes</title>
<h1>Notes App</h1>
<a href="{{ url_for('add_note') }}">Add Note</a>
<hr>
{% for note in notes %}
    <div style="border:1px solid #888; margin-bottom:10px; padding:10px;">
        <h2>{{ note['title'] }}</h2>
        <pre>{{ note['content'] }}</pre>
        <a href="{{ url_for('edit_note', note_id=note['id']) }}">Edit</a>
    </div>
{% else %}
    <p>No notes found.</p>
{% endfor %}
"""

form_layout = """
<!doctype html>
<title>{{ mode }} Note</title>
<h1>{{ mode }} Note</h1>
<form method="post">
    <label>Title:<br>
        <input type="text" name="title" value="{{ title|default('') }}" required>
    </label><br><br>
    <label>Content:<br>
        <textarea name="content" rows="6" cols="40" required>{{ content|default('') }}</textarea>
    </label><br><br>
    <button type="submit">{{ mode }}</button>
    <a href="{{ url_for('index') }}">Cancel</a>
</form>
"""

@app.route('/')
def index():
    return render_template_string(layout, notes=notes)

@app.route('/add', methods=['GET', 'POST'])
def add_note():
    global note_id_counter
    if request.method == 'POST':
        title = request.form.get('title', '')
        content = request.form.get('content', '')
        note = {'id': note_id_counter, 'title': title, 'content': content}
        notes.append(note)
        note_id_counter += 1
        return redirect(url_for('index'))
    return render_template_string(form_layout, mode='Add')

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_note(note_id):
    note = next((n for n in notes if n['id'] == note_id), None)
    if not note:
        return 'Note not found', 404
    if request.method == 'POST':
        note['title'] = request.form.get('title', '')
        note['content'] = request.form.get('content', '')
        return redirect(url_for('index'))
    return render_template_string(form_layout, mode='Edit', title=note['title'], content=note['content'])

if __name__ == '__main__':
    app.run(debug=True)