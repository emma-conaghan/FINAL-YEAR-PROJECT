import os
import json
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

NOTES_FILE = "notes.json"

def load_notes():
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_notes(notes):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)

def get_html_header(title="Notes App"):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: Arial, sans-serif; background: #f0f2f5; color: #333; }}
        .container {{ max-width: 900px; margin: 30px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; margin-bottom: 20px; }}
        h2 {{ color: #34495e; margin-bottom: 15px; }}
        .btn {{ display: inline-block; padding: 8px 16px; background: #3498db; color: white;
                text-decoration: none; border: none; border-radius: 4px; cursor: pointer;
                font-size: 14px; margin: 4px; }}
        .btn:hover {{ background: #2980b9; }}
        .btn-danger {{ background: #e74c3c; }}
        .btn-danger:hover {{ background: #c0392b; }}
        .btn-success {{ background: #27ae60; }}
        .btn-success:hover {{ background: #219a52; }}
        .btn-secondary {{ background: #95a5a6; }}
        .btn-secondary:hover {{ background: #7f8c8d; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .note-card {{ border-left: 4px solid #3498db; }}
        .note-title {{ font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 8px; }}
        .note-date {{ font-size: 12px; color: #7f8c8d; margin-bottom: 10px; }}
        .note-content {{ color: #555; white-space: pre-wrap; margin-bottom: 15px; line-height: 1.5; }}
        .form-group {{ margin-bottom: 15px; }}
        label {{ display: block; margin-bottom: 5px; font-weight: bold; color: #34495e; }}
        input[type=text], textarea {{ width: 100%; padding: 10px; border: 1px solid #ddd;
                                       border-radius: 4px; font-size: 14px; font-family: Arial, sans-serif; }}
        textarea {{ height: 200px; resize: vertical; }}
        input[type=text]:focus, textarea:focus {{ outline: none; border-color: #3498db; }}
        .nav {{ background: #2c3e50; padding: 15px 20px; margin-bottom: 30px; }}
        .nav a {{ color: white; text-decoration: none; font-size: 18px; font-weight: bold; }}
        .nav a:hover {{ color: #3498db; }}
        .actions {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .empty-state {{ text-align: center; color: #7f8c8d; padding: 40px; }}
        .top-bar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
    </style>
</head>
<body>
<div class="nav">
    <a href="/">&#128221; My Notes</a>
</div>
<div class="container">
"""

def get_html_footer():
    return "</div></body></html>"

def render_home(notes):
    html = get_html_header()
    html += """<div class="top-bar">
        <h1>My Notes</h1>
        <a href="/new" class="btn btn-success">+ New Note</a>
    </div>"""
    
    if not notes:
        html += """<div class="card empty-state">
            <p>No notes yet. Create your first note!</p>
            <br>
            <a href="/new" class="btn btn-success">+ New Note</a>
        </div>"""
    else:
        sorted_notes = sorted(notes.items(), key=lambda x: x[1].get("updated", ""), reverse=True)
        for note_id, note in sorted_notes:
            title = escape_html(note.get("title", "Untitled"))
            content = escape_html(note.get("content", ""))
            updated = note.get("updated", "")
            preview = content[:200] + ("..." if len(content) > 200 else "")
            html += f"""<div class="card note-card">
                <div class="note-title">{title}</div>
                <div class="note-date">Last updated: {updated}</div>
                <div class="note-content">{preview}</div>
                <div class="actions">
                    <a href="/view/{note_id}" class="btn">View</a>
                    <a href="/edit/{note_id}" class="btn">Edit</a>
                    <a href="/delete/{note_id}" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</a>
                </div>
            </div>"""
    
    html += get_html_footer()
    return html

def render_new_form(error="", title_val="", content_val=""):
    html = get_html_header("New Note")
    html += "<h2>Create New Note</h2>"
    if error:
        html += f'<div class="card" style="border-left: 4px solid #e74c3c; color: #e74c3c;">{escape_html(error)}</div>'
    html += f"""<div class="card">
        <form method="POST" action="/new">
            <div class="form-group">
                <label for="title">Title</label>
                <input type="text" id="title" name="title" value="{escape_html(title_val)}" placeholder="Enter note title..." required>
            </div>
            <div class="form-group">
                <label for="content">Content</label>
                <textarea id="content" name="content" placeholder="Write your note here...">{escape_html(content_val)}</textarea>
            </div>
            <div class="actions">
                <button type="submit" class="btn btn-success">Save Note</button>
                <a href="/" class="btn btn-secondary">Cancel</a>
            </div>
        </form>
    </div>"""
    html += get_html_footer()
    return html

def render_edit_form(note_id, note, error=""):
    html = get_html_header("Edit Note")
    html += "<h2>Edit Note</h2>"
    if error:
        html += f'<div class="card" style="border-left: 4px solid #e74c3c; color: #e74c3c;">{escape_html(error)}</div>'
    title_val = escape_html(note.get("title", ""))
    content_val = escape_html(note.get("content", ""))
    html += f"""<div class="card">
        <form method="POST" action="/edit/{note_id}">
            <div class="form-group">
                <label for="title">Title</label>
                <input type="text" id="title" name="title" value="{title_val}" placeholder="Enter note title..." required>
            </div>
            <div class="form-group">
                <label for="content">Content</label>
                <textarea id="content" name="content" placeholder="Write your note here...">{content_val}</textarea>
            </div>
            <div class="actions">
                <button type="submit" class="btn btn-success">Update Note</button>
                <a href="/view/{note_id}" class="btn btn-secondary">Cancel</a>
            </div>
        </form>
    </div>"""
    html += get_html_footer()
    return html

def render_view(note_id, note):
    html = get_html_header(note.get("title", "Note"))
    title = escape_html(note.get("title", "Untitled"))
    content = escape_html(note.get("content", ""))
    updated = note.get("updated", "")
    created = note.get("created", "")
    html += f"""<div class="top-bar">
        <h2>{title}</h2>
        <div class="actions">
            <a href="/edit/{note_id}" class="btn">Edit</a>
            <a href="/" class="btn btn-secondary">Back</a>
        </div>
    </div>
    <div class="card note-card">
        <div class="note-date">Created: {created} &nbsp;|&nbsp; Last updated: {updated}</div>
        <div class="note-content">{content}</div>
    </div>"""
    html += get_html_footer()
    return html

def render_404():
    html = get_html_header("Not Found")
    html += """<div class="card empty-state">
        <h2>404 - Page Not Found</h2>
        <br>
        <a href="/" class="btn">Go Home</a>
    </div>"""
    html += get_html_footer()
    return html

def escape_html(text):
    if not isinstance(text, str):
        text = str(text)
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))

def parse_post_body(handler):
    content_length = int(handler.headers.get("Content-Length", 0))
    body = handler.rfile.read(content_length).decode("utf-8")
    return parse_qs(body)

class NotesHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")

    def send_html(self, content, status=200):
        encoded = content.encode("utf-8")
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
        path = parsed.path.rstrip("/") or "/"
        notes = load_notes()

        if path == "/":
            self.send_html(render_home(notes))

        elif path == "/new":
            self.send_html(render_new_form())

        elif path.startswith("/view/"):
            note_id = path[len("/view/"):]
            if note_id in notes:
                self.send_html(render_view(note_id, notes[note_id]))
            else:
                self.send_html(render_404(), 404)

        elif path.startswith("/edit/"):
            note_id = path[len("/edit/"):]
            if note_id in notes:
                self.send_html(render_edit_form(note_id, notes[note_id]))
            else:
                self.send_html(render_404(), 404)

        elif path.startswith("/delete/"):
            note_id = path[len("/delete/"):]
            if note_id in notes:
                del notes[note_id]
                save_notes(notes)
            self.redirect("/")

        else:
            self.send_html(render_404(), 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        notes = load_notes()

        if path == "/new":
            form = parse_post_body(self)
            title = form.get("title", [""])[0].strip()
            content = form.get("content", [""])[0].strip()

            if not title:
                self.send_html(render_new_form(error="Title is required.", title_val=title, content_val=content))
                return

            note_id = str(uuid.uuid4())
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes[note_id] = {
                "title": title,
                "content": content,
                "created": now,
                "updated": now
            }
            save_notes(notes)
            self.redirect(f"/view/{note_id}")

        elif path.startswith("/edit/"):
            note_id = path[len("/edit/"):]
            if note_id not in notes:
                self.send_html(render_404(), 404)
                return

            form = parse_post_body(self)
            title = form.get("title", [""])[0].strip()
            content = form.get("content", [""])[0].strip()

            if not title:
                self.send_html(render_edit_form(note_id, {"title": title, "content": content}, error="Title is required."))
                return

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes[note_id]["title"] = title
            notes[note_id]["content"] = content
            notes[note_id]["updated"] = now
            save_notes(notes)
            self.redirect(f"/view/{note_id}")

        else:
            self.send_html(render_404(), 404)

def run():
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), NotesHandler)
    print(f"Notes app running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()

if __name__ == "__main__":
    run()