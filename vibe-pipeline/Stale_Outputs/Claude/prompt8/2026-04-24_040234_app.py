import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import html

NOTES_FILE = "notes.json"

def load_notes():
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r") as f:
            return json.load(f)
    return []

def save_notes(notes):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f)

def get_next_id(notes):
    if not notes:
        return 1
    return max(n["id"] for n in notes) + 1

def render_base(content, title="Notes App"):
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1, h2 {{
            color: #444;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .note-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 16px;
        }}
        .note-card h3 {{
            margin: 0 0 8px 0;
        }}
        .note-card p {{
            margin: 0 0 10px 0;
            white-space: pre-wrap;
        }}
        .note-actions a {{
            margin-right: 12px;
            font-size: 0.9em;
        }}
        .delete-link {{
            color: #cc0000;
        }}
        form {{
            background: white;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 6px;
        }}
        input[type="text"], textarea {{
            width: 100%;
            padding: 8px;
            margin: 6px 0 14px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 1em;
        }}
        textarea {{
            height: 200px;
            resize: vertical;
        }}
        input[type="submit"] {{
            background-color: #0066cc;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
        }}
        input[type="submit"]:hover {{
            background-color: #0052a3;
        }}
        .nav {{
            margin-bottom: 20px;
        }}
        .empty-msg {{
            color: #888;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">&#8962; All Notes</a> |
        <a href="/new">&#43; New Note</a>
    </div>
    {content}
</body>
</html>"""

def render_home(notes):
    if not notes:
        notes_html = '<p class="empty-msg">No notes yet. <a href="/new">Create your first note!</a></p>'
    else:
        items = []
        for note in reversed(notes):
            note_id = note["id"]
            title = html.escape(note["title"])
            content = html.escape(note["content"])
            items.append(f"""
            <div class="note-card">
                <h3>{title}</h3>
                <p>{content}</p>
                <div class="note-actions">
                    <a href="/view?id={note_id}">View</a>
                    <a href="/edit?id={note_id}">Edit</a>
                    <a href="/delete?id={note_id}" class="delete-link" onclick="return confirm('Delete this note?')">Delete</a>
                </div>
            </div>""")
        notes_html = "".join(items)

    content = f"<h1>My Notes</h1>{notes_html}"
    return render_base(content)

def render_new_form(error=""):
    error_html = f'<p style="color:red;">{html.escape(error)}</p>' if error else ""
    content = f"""
    <h1>New Note</h1>
    {error_html}
    <form method="POST" action="/new">
        <label>Title:</label><br>
        <input type="text" name="title" placeholder="Enter title..." required><br>
        <label>Content:</label><br>
        <textarea name="content" placeholder="Write your note here..."></textarea><br>
        <input type="submit" value="Save Note">
    </form>"""
    return render_base(content, "New Note")

def render_edit_form(note, error=""):
    error_html = f'<p style="color:red;">{html.escape(error)}</p>' if error else ""
    title = html.escape(note["title"])
    content_val = html.escape(note["content"])
    note_id = note["id"]
    content = f"""
    <h1>Edit Note</h1>
    {error_html}
    <form method="POST" action="/edit">
        <input type="hidden" name="id" value="{note_id}">
        <label>Title:</label><br>
        <input type="text" name="title" value="{title}" required><br>
        <label>Content:</label><br>
        <textarea name="content">{content_val}</textarea><br>
        <input type="submit" value="Update Note">
    </form>"""
    return render_base(content, f"Edit: {note['title']}")

def render_view(note):
    title = html.escape(note["title"])
    content_val = html.escape(note["content"])
    note_id = note["id"]
    content = f"""
    <h1>{title}</h1>
    <div class="note-card">
        <p>{content_val}</p>
        <div class="note-actions">
            <a href="/edit?id={note_id}">Edit</a>
            <a href="/delete?id={note_id}" class="delete-link" onclick="return confirm('Delete this note?')">Delete</a>
        </div>
    </div>"""
    return render_base(content, note["title"])

def render_404():
    content = "<h1>404 - Page Not Found</h1><p><a href='/'>Go back home</a></p>"
    return render_base(content)

def render_not_found_note():
    content = "<h1>Note Not Found</h1><p>The note you are looking for does not exist. <a href='/'>Go back home</a></p>"
    return render_base(content)

class NotesHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def send_html(self, html_content, status=200):
        encoded = html_content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "":
            notes = load_notes()
            self.send_html(render_home(notes))

        elif path == "/new":
            self.send_html(render_new_form())

        elif path == "/edit":
            note_id = params.get("id", [None])[0]
            if note_id is None:
                self.send_html(render_not_found_note(), 404)
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.send_html(render_not_found_note(), 404)
                return
            notes = load_notes()
            note = next((n for n in notes if n["id"] == note_id), None)
            if note is None:
                self.send_html(render_not_found_note(), 404)
                return
            self.send_html(render_edit_form(note))

        elif path == "/view":
            note_id = params.get("id", [None])[0]
            if note_id is None:
                self.send_html(render_not_found_note(), 404)
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.send_html(render_not_found_note(), 404)
                return
            notes = load_notes()
            note = next((n for n in notes if n["id"] == note_id), None)
            if note is None:
                self.send_html(render_not_found_note(), 404)
                return
            self.send_html(render_view(note))

        elif path == "/delete":
            note_id = params.get("id", [None])[0]
            if note_id is None:
                self.redirect("/")
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.redirect("/")
                return
            notes = load_notes()
            notes = [n for n in notes if n["id"] != note_id]
            save_notes(notes)
            self.redirect("/")

        else:
            self.send_html(render_404(), 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        form_data = parse_qs(body)

        if path == "/new":
            title = form_data.get("title", [""])[0].strip()
            content = form_data.get("content", [""])[0].strip()

            if not title:
                self.send_html(render_new_form(error="Title is required."))
                return

            notes = load_notes()
            new_note = {
                "id": get_next_id(notes),
                "title": title,
                "content": content
            }
            notes.append(new_note)
            save_notes(notes)
            self.redirect("/")

        elif path == "/edit":
            note_id_str = form_data.get("id", [""])[0].strip()
            title = form_data.get("title", [""])[0].strip()
            content = form_data.get("content", [""])[0].strip()

            try:
                note_id = int(note_id_str)
            except ValueError:
                self.redirect("/")
                return

            notes = load_notes()
            note = next((n for n in notes if n["id"] == note_id), None)

            if note is None:
                self.send_html(render_not_found_note(), 404)
                return

            if not title:
                self.send_html(render_edit_form(note, error="Title is required."))
                return

            note["title"] = title
            note["content"] = content
            save_notes(notes)
            self.redirect(f"/view?id={note_id}")

        else:
            self.send_html(render_404(), 404)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), NotesHandler)
    print(f"Notes app running at http://localhost:{port}")
    server.serve_forever()