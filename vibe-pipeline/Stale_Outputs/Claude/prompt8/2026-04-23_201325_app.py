from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import json
import os
import uuid
from datetime import datetime

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
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1, h2 {{ color: #333; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .btn {{ display: inline-block; padding: 8px 16px; background: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; font-size: 14px; }}
        .btn:hover {{ background: #0052a3; text-decoration: none; color: white; }}
        .btn-danger {{ background: #cc0000; }}
        .btn-danger:hover {{ background: #a30000; }}
        .btn-secondary {{ background: #666; }}
        .btn-secondary:hover {{ background: #444; }}
        .note-card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .note-title {{ font-size: 1.2em; font-weight: bold; color: #333; margin-bottom: 5px; }}
        .note-meta {{ font-size: 0.85em; color: #888; margin-bottom: 10px; }}
        .note-preview {{ color: #555; white-space: pre-wrap; word-break: break-word; }}
        .note-actions {{ margin-top: 10px; }}
        .note-actions a {{ margin-right: 10px; }}
        form {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        label {{ display: block; margin-bottom: 5px; font-weight: bold; color: #333; }}
        input[type=text], textarea {{ width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-size: 14px; font-family: Arial, sans-serif; }}
        textarea {{ height: 200px; resize: vertical; }}
        .form-group {{ margin-bottom: 15px; }}
        .nav {{ margin-bottom: 20px; }}
        .empty {{ text-align: center; color: #888; padding: 40px; }}
        .full-content {{ white-space: pre-wrap; word-break: break-word; color: #333; line-height: 1.6; }}
        .flash {{ padding: 10px 15px; background: #d4edda; color: #155724; border-radius: 4px; margin-bottom: 15px; border: 1px solid #c3e6cb; }}
    </style>
</head>
<body>
"""

def get_html_footer():
    return """</body>
</html>"""

def render_index(notes, message=None):
    html = get_html_header("My Notes")
    html += "<h1>📝 My Notes</h1>\n"
    html += '<div class="nav"><a href="/new" class="btn">+ New Note</a></div>\n'
    if message:
        html += f'<div class="flash">{message}</div>\n'
    if not notes:
        html += '<div class="empty"><p>No notes yet. Create your first note!</p></div>\n'
    else:
        sorted_notes = sorted(notes.values(), key=lambda x: x.get("updated_at", ""), reverse=True)
        for note in sorted_notes:
            preview = note["content"][:200] + ("..." if len(note["content"]) > 200 else "")
            note_id = note["id"]
            created = note.get("created_at", "")[:16].replace("T", " ")
            updated = note.get("updated_at", "")[:16].replace("T", " ")
            html += f"""<div class="note-card">
    <div class="note-title">{escape_html(note['title'])}</div>
    <div class="note-meta">Created: {created} | Updated: {updated}</div>
    <div class="note-preview">{escape_html(preview)}</div>
    <div class="note-actions">
        <a href="/view?id={note_id}" class="btn">View</a>
        <a href="/edit?id={note_id}" class="btn btn-secondary">Edit</a>
        <a href="/delete?id={note_id}" class="btn btn-danger" onclick="return confirm('Delete this note?')">Delete</a>
    </div>
</div>\n"""
    html += get_html_footer()
    return html

def render_new_form(title="", content="", error=""):
    html = get_html_header("New Note")
    html += '<div class="nav"><a href="/">&larr; Back to Notes</a></div>\n'
    html += "<h1>New Note</h1>\n"
    if error:
        html += f'<div class="flash" style="background:#f8d7da;color:#721c24;border-color:#f5c6cb;">{error}</div>\n'
    html += f"""<form method="POST" action="/new">
    <div class="form-group">
        <label for="title">Title</label>
        <input type="text" id="title" name="title" value="{escape_html(title)}" placeholder="Note title..." required>
    </div>
    <div class="form-group">
        <label for="content">Content</label>
        <textarea id="content" name="content" placeholder="Write your note here...">{escape_html(content)}</textarea>
    </div>
    <button type="submit" class="btn">Save Note</button>
    <a href="/" class="btn btn-secondary" style="margin-left:10px;">Cancel</a>
</form>\n"""
    html += get_html_footer()
    return html

def render_edit_form(note, error=""):
    html = get_html_header(f"Edit: {note['title']}")
    html += '<div class="nav"><a href="/">&larr; Back to Notes</a></div>\n'
    html += f"<h1>Edit Note</h1>\n"
    if error:
        html += f'<div class="flash" style="background:#f8d7da;color:#721c24;border-color:#f5c6cb;">{error}</div>\n'
    html += f"""<form method="POST" action="/edit">
    <input type="hidden" name="id" value="{note['id']}">
    <div class="form-group">
        <label for="title">Title</label>
        <input type="text" id="title" name="title" value="{escape_html(note['title'])}" required>
    </div>
    <div class="form-group">
        <label for="content">Content</label>
        <textarea id="content" name="content">{escape_html(note['content'])}</textarea>
    </div>
    <button type="submit" class="btn">Update Note</button>
    <a href="/" class="btn btn-secondary" style="margin-left:10px;">Cancel</a>
</form>\n"""
    html += get_html_footer()
    return html

def render_view(note):
    html = get_html_header(note["title"])
    html += '<div class="nav"><a href="/">&larr; Back to Notes</a></div>\n'
    created = note.get("created_at", "")[:16].replace("T", " ")
    updated = note.get("updated_at", "")[:16].replace("T", " ")
    html += f"""<div class="note-card">
    <h2>{escape_html(note['title'])}</h2>
    <div class="note-meta">Created: {created} | Last updated: {updated}</div>
    <div class="full-content">{escape_html(note['content'])}</div>
    <div class="note-actions" style="margin-top:20px;">
        <a href="/edit?id={note['id']}" class="btn btn-secondary">Edit</a>
        <a href="/delete?id={note['id']}" class="btn btn-danger" style="margin-left:10px;" onclick="return confirm('Delete this note?')">Delete</a>
    </div>
</div>\n"""
    html += get_html_footer()
    return html

def render_404():
    html = get_html_header("Not Found")
    html += "<h1>404 - Page Not Found</h1>\n"
    html += '<p><a href="/">Go back to notes</a></p>\n'
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
            .replace("'", "&#x27;"))

def parse_post_data(handler):
    content_length = int(handler.headers.get("Content-Length", 0))
    post_data = handler.rfile.read(content_length).decode("utf-8")
    return urllib.parse.parse_qs(post_data, keep_blank_values=True)

class NotesHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

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
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        notes = load_notes()

        if path == "/" or path == "":
            self.send_html(render_index(notes))

        elif path == "/new":
            self.send_html(render_new_form())

        elif path == "/view":
            note_id = query.get("id", [None])[0]
            if note_id and note_id in notes:
                self.send_html(render_view(notes[note_id]))
            else:
                self.send_html(render_404(), 404)

        elif path == "/edit":
            note_id = query.get("id", [None])[0]
            if note_id and note_id in notes:
                self.send_html(render_edit_form(notes[note_id]))
            else:
                self.send_html(render_404(), 404)

        elif path == "/delete":
            note_id = query.get("id", [None])[0]
            if note_id and note_id in notes:
                del notes[note_id]
                save_notes(notes)
                self.redirect("/?msg=deleted")
            else:
                self.send_html(render_404(), 404)

        else:
            self.send_html(render_404(), 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        notes = load_notes()

        if path == "/new":
            data = parse_post_data(self)
            title = data.get("title", [""])[0].strip()
            content = data.get("content", [""])[0].strip()

            if not title:
                self.send_html(render_new_form(title=title, content=content, error="Title is required."))
                return

            note_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            notes[note_id] = {
                "id": note_id,
                "title": title,
                "content": content,
                "created_at": now,
                "updated_at": now
            }
            save_notes(notes)
            self.redirect("/")

        elif path == "/edit":
            data = parse_post_data(self)
            note_id = data.get("id", [""])[0].strip()
            title = data.get("title", [""])[0].strip()
            content = data.get("content", [""])[0].strip()

            if note_id not in notes:
                self.send_html(render_404(), 404)
                return

            if not title:
                note = notes[note_id]
                note["title"] = title
                note["content"] = content
                self.send_html(render_edit_form(note, error="Title is required."))
                return

            notes[note_id]["title"] = title
            notes[note_id]["content"] = content
            notes[note_id]["updated_at"] = datetime.now().isoformat()
            save_notes(notes)
            self.redirect(f"/view?id={note_id}")

        else:
            self.send_html(render_404(), 404)

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