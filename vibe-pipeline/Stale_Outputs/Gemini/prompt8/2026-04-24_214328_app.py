from flask import Flask, request, redirect, render_template_string

app = Flask(__name__)

notes = []

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Simple Notes</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
        .note { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 4px; }
        .note h3 { margin-top: 0; }
        form input, form textarea { width: 100%; margin-bottom: 10px; display: block; padding: 8px; }
        form textarea { height: 100px; }
        .nav { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">Home</a> | <a href="/create">New Note</a>
    </div>
    {% block content %}{% endblock %}
</body>
</html>
"""

INDEX_HTML = """
{% extends "base" %}
{% block content %}
    <h1>My Notes</h1>
    {% if not notes %}
        <p>No notes found.</p>
    {% endif %}
    {% for note in notes %}
        <div class="note">
            <h3>{{ note.title }}</h3>
            <p>{{ note.content }}</p>
            <a href="/edit/{{ loop.index0 }}">Edit</a>
        </div>
    {% endfor %}
{% endblock %}
"""

FORM_HTML = """
{% extends "base" %}
{% block content %}
    <h1>{{ action }} Note</h1>
    <form method="POST">
        <input type="text" name="title" placeholder="Title" value="{{ note.title if note else '' }}" required>
        <textarea name="content" placeholder="Content" required>{{ note.content if note else '' }}</textarea>
        <button type="submit">Save Note</button>
    </form>
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, notes=notes), 200, {'Content-Type': 'text/html'}

@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')
        notes.append({'title': title, 'content': content})
        return redirect('/')
    return render_template_string(FORM_HTML, action="Create", note=None)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit(note_id):
    if note_id < 0 or note_id >= len(notes):
        return redirect('/')
    
    if request.method == 'POST':
        notes[note_id]['title'] = request.form.get('title')
        notes[note_id]['content'] = request.form.get('content')
        return redirect('/')
    
    return render_template_string(FORM_HTML, action="Edit", note=notes[note_id])

@app.context_processor
def inject_base():
    return {'base_html': BASE_HTML}

# Overriding render_template_string slightly to handle the base template inheritance simulation
def smart_render(template_str, **context):
    full_template = "{% filter rstrip %}" + BASE_HTML.replace('{% block content %}{% endblock %}', template_str.split('{% block content %}')[-1].split('{% endblock %}')[0]) + "{% endfilter %}"
    return render_template_string(full_template, **context)

# Re-defining routes to use a simpler flat structure since Jinja2 inheritance 
# usually requires physical files.
@app.route('/')
def index_flat():
    html = BASE_HTML.replace('{% block content %}{% endblock %}', INDEX_HTML.split('{% block content %}')[-1].split('{% endblock %}')[0])
    return render_template_string(html, notes=notes)

@app.route('/create', methods=['GET', 'POST'])
def create_flat():
    if request.method == 'POST':
        notes.append({'title': request.form.get('title'), 'content': request.form.get('content')})
        return redirect('/')
    html = BASE_HTML.replace('{% block content %}{% endblock %}', FORM_HTML.split('{% block content %}')[-1].split('{% endblock %}')[0])
    return render_template_string(html, action="Create", note=None)

@app.route('/edit/<int:note_id>', methods=['GET', 'POST'])
def edit_flat(note_id):
    if note_id < 0 or note_id >= len(notes):
        return redirect('/')
    if request.method == 'POST':
        notes[note_id] = {'title': request.form.get('title'), 'content': request.form.get('content')}
        return redirect('/')
    html = BASE_HTML.replace('{% block content %}{% endblock %}', FORM_HTML.split('{% block content %}')[-1].split('{% endblock %}')[0])
    return render_template_string(html, action="Edit", note=notes[note_id])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)