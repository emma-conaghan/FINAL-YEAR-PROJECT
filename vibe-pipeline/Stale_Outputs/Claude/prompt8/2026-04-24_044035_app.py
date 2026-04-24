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

def get_html_page(title, content):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: Arial, sans-serif; background: #f4f4f4; color: #333; }}
        .container {{ max-width: 900px; margin: 30px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; margin-bottom: 20px; }}
        h2 {{ color: #34495e; margin-bottom: 15px; }}
        .nav {{ background: #2c3e50; padding: 15px; margin-bottom: 30px; }}
        .nav a {{ color: white; text-decoration: none; margin-right: 20px; font-size: 16px; }}
        .nav a:hover {{ text-decoration: underline; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .note-card {{ border-left: 4px solid #3498db; }}
        .note-title {{ font-size: 20px; font-weight: bold; margin-bottom: 8px; color: #2c3e50; }}
        .note-date {{ font-size: 12px; color: #999; margin-bottom: 10px; }}
        .note-content {{ white-space: pre-wrap; line-height: 1.6; }}
        .btn {{ display: inline-block; padding: 8px 16px; border-radius: 4px; text-decoration: none; color: white; border: none; cursor: pointer; font-size: 14px; margin-right: 8px; }}
        .btn-primary {{ background: #3498db; }}
        .btn-success {{ background: #27ae60; }}
        .btn-danger {{ background: #e74c3c; }}
        .btn-warning {{ background: #f39c12; }}
        .btn:hover {{ opacity: 0.85; }}
        form label {{ display: block; margin-bottom: 5px; font-weight: bold; color: #555; }}
        form input[type=text], form textarea {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 15px; margin-bottom: 15px; }}
        form textarea {{ height: 200px; resize: vertical; font-family: Arial, sans-serif; }}
        .actions {{ margin-top: 12px; }}
        .empty-msg {{ color: #999; font-style: italic; }}
        .success {{ background: #d4edda; color: #155724; padding: 10px; border-radius: 4px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">📝 My Notes</a>
        <a href="/new">+ New Note</a>
    </div>
    <div class="container">
        {content}
    </div>
</body>
</html>"""

def render_notes_list(notes, message=""):
    msg_html = f'<div class="success">{message}</div>' if message else ""
    if not notes:
        notes_html = '<p class="empty-msg">No notes yet. <a href="/new">Create your first note!</a></p>'
    else:
        sorted_notes = sorted(notes.values(), key=lambda x: x.get("updated", ""), reverse=True)
        notes_html = ""
        for note in sorted_notes:
            safe_title = note["title"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            preview = note["content"][:150].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if len(note["content"]) > 150:
                preview += "..."
            notes_html += f"""
            <div class="card note-card">
                <div class="note-title">{safe_title}</div>
                <div class="note-date">Last updated: {note.get('updated', 'Unknown')}</div>
                <div class="note-content">{preview}</div>
                <div class="actions">
                    <a href="/view?id={note['id']}" class="btn btn-primary">View</a>
                    <a href="/edit?id={note['id']}" class="btn btn-warning">Edit</a>
                    <a href="/delete?id={note['id']}" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</a>
                </div>
            </div>"""
    content = f"""
        <h1>📝 My Notes</h1>
        {msg_html}
        {notes_html}
    """
    return get_html_page("My Notes", content)

def render_new_note_form(error=""):
    err_html = f'<div class="success" style="background:#f8d7da;color:#721c24;">{error}</div>' if error else ""
    content = f"""
        <h2>Create New Note</h2>
        <div class="card">
            {err_html}
            <form method="POST" action="/create">
                <label for="title">Title</label>
                <input type="text" id="title" name="title" placeholder="Enter note title..." required>
                <label for="content">Content</label>
                <textarea id="content" name="content" placeholder="Write your note here..."></textarea>
                <button type="submit" class="btn btn-success">Save Note</button>
                <a href="/" class="btn btn-primary">Cancel</a>
            </form>
        </div>
    """
    return get_html_page("New Note", content)

def render_view_note(note):
    safe_title = note["title"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_content = note["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    content = f"""
        <h2>{safe_title}</h2>
        <div class="card note-card">
            <div class="note-date">Created: {note.get('created', 'Unknown')} | Updated: {note.get('updated', 'Unknown')}</div>
            <div class="note-content" style="margin-top:10px;">{safe_content}</div>
            <div class="actions" style="margin-top:20px;">
                <a href="/edit?id={note['id']}" class="btn btn-warning">Edit</a>
                <a href="/" class="btn btn-primary">Back to Notes</a>
                <a href="/delete?id={note['id']}" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</a>
            </div>
        </div>
    """
    return get_html_page(safe_title, content)

def render_edit_note_form(note, error=""):
    err_html = f'<div class="success" style="background:#f8d7da;color:#721c24;">{error}</div>' if error else ""
    safe_title = note["title"].replace('"', '&quot;')
    safe_content = note["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    content = f"""
        <h2>Edit Note</h2>
        <div class="card">
            {err_html}
            <form method="POST" action="/update">
                <input type="hidden" name="id" value="{note['id']}">
                <label for="title">Title</label>
                <input type="text" id="title" name="title" value="{safe_title}" required>
                <label for="content">Content</label>
                <textarea id="content" name="content">{safe_content}</textarea>
                <button type="submit" class="btn btn-success">Update Note</button>
                <a href="/view?id={note['id']}" class="btn btn-primary">Cancel</a>
            </form>
        </div>
    """
    return get_html_page("Edit Note", content)

def render_404():
    content = '<h2>404 - Page Not Found</h2><p><a href="/">Go back home</a></p>'
    return get_html_page("Not Found", content)

def render_note_not_found():
    content = '<h2>Note Not Found</h2><p>The note you are looking for does not exist. <a href="/">Go back home</a></p>'
    return get_html_page("Note Not Found", content)

class NotesHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_html(self, html, status=200):
        encoded = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode("utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        notes = load_notes()

        if path == "/":
            self.send_html(render_notes_list(notes))
        elif path == "/new":
            self.send_html(render_new_note_form())
        elif path == "/view":
            note_id = params.get("id", [None])[0]
            if note_id and note_id in notes:
                self.send_html(render_view_note(notes[note_id]))
            else:
                self.send_html(render_note_not_found(), 404)
        elif path == "/edit":
            note_id = params.get("id", [None])[0]
            if note_id and note_id in notes:
                self.send_html(render_edit_note_form(notes[note_id]))
            else:
                self.send_html(render_note_not_found(), 404)
        elif path == "/delete":
            note_id = params.get("id", [None])[0]
            if note_id and note_id in notes:
                del notes[note_id]
                save_notes(notes)
                self.redirect("/?deleted=1")
            else:
                self.send_html(render_note_not_found(), 404)
        else:
            self.send_html(render_404(), 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self.read_body()
        data = parse_qs(body)

        notes = load_notes()

        if path == "/create":
            title = data.get("title", [""])[0].strip()
            content = data.get("content", [""])[0].strip()
            if not title:
                self.send_html(render_new_note_form(error="Title is required."))
                return
            note_id = str(uuid.uuid4())
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes[note_id] = {
                "id": note_id,
                "title": title,
                "content": content,
                "created": now,
                "updated": now
            }
            save_notes(notes)
            self.redirect(f"/view?id={note_id}")
        elif path == "/update":
            note_id = data.get("id", [""])[0].strip()
            title = data.get("title", [""])[0].strip()
            content = data.get("content", [""])[0].strip()
            if note_id not in notes:
                self.send_html(render_note_not_found(), 404)
                return
            if not title:
                self.send_html(render_edit_note_form(notes[note_id], error="Title is required."))
                return
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes[note_id]["title"] = title
            notes[note_id]["content"] = content
            notes[note_id]["updated"] = now
            save_notes(notes)
            self.redirect(f"/view?id={note_id}")
        else:
            self.send_html(render_404(), 404)

def run():
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), NotesHandler)
    print(f"Notes app running at http://{host}:{port}")
    server.serve_forever()

if __name__ == "__main__":
    run()