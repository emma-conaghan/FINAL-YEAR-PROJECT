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
        body {{ font-family: Arial, sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 900px; margin: 30px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; margin-bottom: 20px; }}
        h2 {{ color: #34495e; margin-bottom: 15px; }}
        nav {{ background: #2c3e50; padding: 15px 20px; }}
        nav a {{ color: white; text-decoration: none; margin-right: 20px; font-size: 16px; }}
        nav a:hover {{ text-decoration: underline; }}
        .note-card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px;
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .note-card h3 {{ color: #2c3e50; margin-bottom: 8px; }}
        .note-card .meta {{ color: #888; font-size: 13px; margin-bottom: 10px; }}
        .note-card .content {{ color: #555; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
        .note-card .actions {{ margin-top: 12px; }}
        .btn {{ display: inline-block; padding: 8px 16px; border-radius: 5px; text-decoration: none;
               font-size: 14px; cursor: pointer; border: none; margin-right: 8px; }}
        .btn-primary {{ background: #3498db; color: white; }}
        .btn-primary:hover {{ background: #2980b9; }}
        .btn-danger {{ background: #e74c3c; color: white; }}
        .btn-danger:hover {{ background: #c0392b; }}
        .btn-secondary {{ background: #95a5a6; color: white; }}
        .btn-secondary:hover {{ background: #7f8c8d; }}
        form {{ background: white; border-radius: 8px; padding: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        label {{ display: block; margin-bottom: 6px; font-weight: bold; color: #2c3e50; }}
        input[type=text], textarea {{
            width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px;
            font-size: 15px; margin-bottom: 15px; font-family: Arial, sans-serif;
        }}
        textarea {{ height: 200px; resize: vertical; }}
        input[type=text]:focus, textarea:focus {{ outline: none; border-color: #3498db; }}
        .empty {{ text-align: center; color: #888; padding: 40px; font-size: 18px; }}
        .flash {{ background: #2ecc71; color: white; padding: 12px 20px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
<nav>
    <a href="/">📝 My Notes</a>
    <a href="/new">+ New Note</a>
</nav>
<div class="container">
"""

def get_html_footer():
    return """</div></body></html>"""

def render_home(notes, flash=None):
    html = get_html_header("My Notes")
    html += "<h1>My Notes</h1>"
    if flash:
        html += f'<div class="flash">{flash}</div>'
    if not notes:
        html += '<div class="empty">No notes yet. <a href="/new">Create your first note!</a></div>'
    else:
        sorted_notes = sorted(notes.items(), key=lambda x: x[1].get("updated", ""), reverse=True)
        for note_id, note in sorted_notes:
            title = note.get("title", "Untitled")
            content = note.get("content", "")
            created = note.get("created", "")
            updated = note.get("updated", "")
            preview = content[:200] + ("..." if len(content) > 200 else "")
            html += f"""
<div class="note-card">
    <h3>{title}</h3>
    <div class="meta">Created: {created} | Updated: {updated}</div>
    <div class="content">{preview}</div>
    <div class="actions">
        <a href="/view?id={note_id}" class="btn btn-secondary">View</a>
        <a href="/edit?id={note_id}" class="btn btn-primary">Edit</a>
        <a href="/delete?id={note_id}" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</a>
    </div>
</div>"""
    html += get_html_footer()
    return html

def render_new_note():
    html = get_html_header("New Note")
    html += """
<h1>New Note</h1>
<form method="POST" action="/create">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" placeholder="Enter note title..." required>
    <label for="content">Content</label>
    <textarea id="content" name="content" placeholder="Write your note here..."></textarea>
    <button type="submit" class="btn btn-primary">Save Note</button>
    <a href="/" class="btn btn-secondary">Cancel</a>
</form>"""
    html += get_html_footer()
    return html

def render_edit_note(note_id, note):
    title = note.get("title", "")
    content = note.get("content", "")
    html = get_html_header("Edit Note")
    html += f"""
<h1>Edit Note</h1>
<form method="POST" action="/update">
    <input type="hidden" name="id" value="{note_id}">
    <label for="title">Title</label>
    <input type="text" id="title" name="title" value="{title}" required>
    <label for="content">Content</label>
    <textarea id="content" name="content">{content}</textarea>
    <button type="submit" class="btn btn-primary">Update Note</button>
    <a href="/" class="btn btn-secondary">Cancel</a>
</form>"""
    html += get_html_footer()
    return html

def render_view_note(note_id, note):
    title = note.get("title", "Untitled")
    content = note.get("content", "")
    created = note.get("created", "")
    updated = note.get("updated", "")
    html = get_html_header(title)
    html += f"""
<h1>{title}</h1>
<div class="note-card">
    <div class="meta">Created: {created} | Updated: {updated}</div>
    <div class="content">{content}</div>
    <div class="actions" style="margin-top:20px;">
        <a href="/" class="btn btn-secondary">Back</a>
        <a href="/edit?id={note_id}" class="btn btn-primary">Edit</a>
        <a href="/delete?id={note_id}" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</a>
    </div>
</div>"""
    html += get_html_footer()
    return html

def render_error(message):
    html = get_html_header("Error")
    html += f'<h1>Error</h1><div class="note-card"><p>{message}</p><a href="/" class="btn btn-secondary">Go Home</a></div>'
    html += get_html_footer()
    return html

class NotesHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {format % args}")

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

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        notes = load_notes()

        if path == "/":
            self.send_html(render_home(notes))

        elif path == "/new":
            self.send_html(render_new_note())

        elif path == "/edit":
            note_id = params.get("id", [None])[0]
            if note_id and note_id in notes:
                self.send_html(render_edit_note(note_id, notes[note_id]))
            else:
                self.send_html(render_error("Note not found."), 404)

        elif path == "/view":
            note_id = params.get("id", [None])[0]
            if note_id and note_id in notes:
                self.send_html(render_view_note(note_id, notes[note_id]))
            else:
                self.send_html(render_error("Note not found."), 404)

        elif path == "/delete":
            note_id = params.get("id", [None])[0]
            if note_id and note_id in notes:
                del notes[note_id]
                save_notes(notes)
                self.redirect("/?flash=deleted")
            else:
                self.send_html(render_error("Note not found."), 404)

        else:
            self.send_html(render_error("Page not found."), 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        params = parse_qs(body)

        def get_param(key):
            values = params.get(key, [""])
            return values[0] if values else ""

        notes = load_notes()
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/create":
            title = get_param("title").strip() or "Untitled"
            content = get_param("content").strip()
            note_id = str(uuid.uuid4())
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notes[note_id] = {
                "title": title,
                "content": content,
                "created": now,
                "updated": now
            }
            save_notes(notes)
            self.redirect("/")

        elif path == "/update":
            note_id = get_param("id").strip()
            if note_id and note_id in notes:
                title = get_param("title").strip() or "Untitled"
                content = get_param("content").strip()
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                notes[note_id]["title"] = title
                notes[note_id]["content"] = content
                notes[note_id]["updated"] = now
                save_notes(notes)
                self.redirect(f"/view?id={note_id}")
            else:
                self.send_html(render_error("Note not found."), 404)

        else:
            self.send_html(render_error("Not found."), 404)

def main():
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), NotesHandler)
    print(f"Notes app running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()

if __name__ == "__main__":
    main()