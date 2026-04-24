import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import html
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

def get_base_html(title, content):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: Arial, sans-serif; background: #f0f2f5; color: #333; }}
        header {{ background: #4a90e2; color: white; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }}
        header h1 {{ font-size: 1.5rem; }}
        header a {{ color: white; text-decoration: none; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 4px; }}
        header a:hover {{ background: rgba(255,255,255,0.3); }}
        .container {{ max-width: 900px; margin: 24px auto; padding: 0 16px; }}
        .card {{ background: white; border-radius: 8px; padding: 24px; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .note-card {{ border-left: 4px solid #4a90e2; }}
        .note-card h3 {{ font-size: 1.2rem; margin-bottom: 8px; color: #4a90e2; }}
        .note-card .meta {{ font-size: 0.8rem; color: #888; margin-bottom: 12px; }}
        .note-card .content {{ white-space: pre-wrap; word-break: break-word; line-height: 1.6; }}
        .note-card .actions {{ margin-top: 16px; display: flex; gap: 8px; }}
        .btn {{ display: inline-block; padding: 8px 16px; border-radius: 4px; border: none; cursor: pointer; font-size: 0.9rem; text-decoration: none; }}
        .btn-primary {{ background: #4a90e2; color: white; }}
        .btn-primary:hover {{ background: #357abd; }}
        .btn-danger {{ background: #e24a4a; color: white; }}
        .btn-danger:hover {{ background: #bd3535; }}
        .btn-secondary {{ background: #888; color: white; }}
        .btn-secondary:hover {{ background: #666; }}
        form label {{ display: block; margin-bottom: 4px; font-weight: bold; font-size: 0.9rem; }}
        form input[type=text], form textarea {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; margin-bottom: 16px; font-family: inherit; }}
        form textarea {{ height: 200px; resize: vertical; }}
        form input[type=text]:focus, form textarea:focus {{ outline: none; border-color: #4a90e2; }}
        .empty {{ text-align: center; color: #888; padding: 48px; }}
        .flash {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px; }}
    </style>
</head>
<body>
    <header>
        <h1>📝 Personal Notes</h1>
        <a href="/new">+ New Note</a>
    </header>
    <div class="container">
        {content}
    </div>
</body>
</html>"""

def render_home(notes, flash=None):
    flash_html = f'<div class="flash">{html.escape(flash)}</div>' if flash else ""
    if not notes:
        content = flash_html + '<div class="card empty"><p>No notes yet. <a href="/new">Create your first note!</a></p></div>'
    else:
        notes_sorted = sorted(notes.values(), key=lambda n: n.get("updated_at", ""), reverse=True)
        cards = ""
        for note in notes_sorted:
            note_id = note["id"]
            title_safe = html.escape(note["title"])
            content_safe = html.escape(note["content"])
            created = note.get("created_at", "")[:16].replace("T", " ")
            updated = note.get("updated_at", "")[:16].replace("T", " ")
            preview = content_safe[:300] + ("..." if len(content_safe) > 300 else "")
            cards += f"""
            <div class="card note-card">
                <h3>{title_safe}</h3>
                <div class="meta">Created: {created} &nbsp;|&nbsp; Updated: {updated}</div>
                <div class="content">{preview}</div>
                <div class="actions">
                    <a href="/view/{note_id}" class="btn btn-secondary">View</a>
                    <a href="/edit/{note_id}" class="btn btn-primary">Edit</a>
                    <form method="post" action="/delete/{note_id}" style="display:inline;" onsubmit="return confirm('Delete this note?')">
                        <button type="submit" class="btn btn-danger">Delete</button>
                    </form>
                </div>
            </div>"""
        content = flash_html + cards
    return get_base_html("Personal Notes", content)

def render_new_form(error=None, title_val="", content_val=""):
    error_html = f'<div class="flash" style="background:#f8d7da;color:#721c24;border-color:#f5c6cb;">{html.escape(error)}</div>' if error else ""
    form = f"""
    <div class="card">
        <h2 style="margin-bottom:20px;">Create New Note</h2>
        {error_html}
        <form method="post" action="/new">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" placeholder="Enter note title..." value="{html.escape(title_val)}" required>
            <label for="content">Content</label>
            <textarea id="content" name="content" placeholder="Write your note here...">{html.escape(content_val)}</textarea>
            <div style="display:flex;gap:8px;">
                <button type="submit" class="btn btn-primary">Save Note</button>
                <a href="/" class="btn btn-secondary">Cancel</a>
            </div>
        </form>
    </div>"""
    return get_base_html("New Note", form)

def render_edit_form(note, error=None):
    error_html = f'<div class="flash" style="background:#f8d7da;color:#721c24;border-color:#f5c6cb;">{html.escape(error)}</div>' if error else ""
    form = f"""
    <div class="card">
        <h2 style="margin-bottom:20px;">Edit Note</h2>
        {error_html}
        <form method="post" action="/edit/{note['id']}">
            <label for="title">Title</label>
            <input type="text" id="title" name="title" placeholder="Enter note title..." value="{html.escape(note['title'])}" required>
            <label for="content">Content</label>
            <textarea id="content" name="content" placeholder="Write your note here...">{html.escape(note['content'])}</textarea>
            <div style="display:flex;gap:8px;">
                <button type="submit" class="btn btn-primary">Update Note</button>
                <a href="/" class="btn btn-secondary">Cancel</a>
            </div>
        </form>
    </div>"""
    return get_base_html("Edit Note", form)

def render_view(note):
    title_safe = html.escape(note["title"])
    content_safe = html.escape(note["content"])
    created = note.get("created_at", "")[:16].replace("T", " ")
    updated = note.get("updated_at", "")[:16].replace("T", " ")
    note_id = note["id"]
    body = f"""
    <div class="card note-card">
        <h2 style="margin-bottom:8px;">{title_safe}</h2>
        <div class="meta" style="margin-bottom:16px;">Created: {created} &nbsp;|&nbsp; Updated: {updated}</div>
        <div class="content" style="font-size:1rem;line-height:1.8;">{content_safe}</div>
        <div class="actions" style="margin-top:20px;">
            <a href="/" class="btn btn-secondary">← Back</a>
            <a href="/edit/{note_id}" class="btn btn-primary">Edit</a>
            <form method="post" action="/delete/{note_id}" style="display:inline;" onsubmit="return confirm('Delete this note?')">
                <button type="submit" class="btn btn-danger">Delete</button>
            </form>
        </div>
    </div>"""
    return get_base_html(note["title"], body)

def render_404():
    body = '<div class="card empty"><h2>404 - Not Found</h2><p><a href="/">Go Home</a></p></div>'
    return get_base_html("Not Found", body)

class NotesHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

    def send_html(self, html_content, status=200):
        encoded = html_content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def redirect(self, location):
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode("utf-8")

    def parse_form(self):
        body = self.read_body()
        return parse_qs(body, keep_blank_values=True)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            notes = load_notes()
            self.send_html(render_home(notes))

        elif path == "/new":
            self.send_html(render_new_form())

        elif path.startswith("/edit/"):
            note_id = path[len("/edit/"):]
            notes = load_notes()
            if note_id not in notes:
                self.send_html(render_404(), status=404)
            else:
                self.send_html(render_edit_form(notes[note_id]))

        elif path.startswith("/view/"):
            note_id = path[len("/view/"):]
            notes = load_notes()
            if note_id not in notes:
                self.send_html(render_404(), status=404)
            else:
                self.send_html(render_view(notes[note_id]))

        else:
            self.send_html(render_404(), status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/new":
            form = self.parse_form()
            title = form.get("title", [""])[0].strip()
            content = form.get("content", [""])[0].strip()
            if not title:
                self.send_html(render_new_form(error="Title is required.", title_val=title, content_val=content))
                return
            notes = load_notes()
            note_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            notes[note_id] = {
                "id": note_id,
                "title": title,
                "content": content,
                "created_at": now,
                "updated_at": now,
            }
            save_notes(notes)
            self.redirect("/")

        elif path.startswith("/edit/"):
            note_id = path[len("/edit/"):]
            notes = load_notes()
            if note_id not in notes:
                self.send_html(render_404(), status=404)
                return
            form = self.parse_form()
            title = form.get("title", [""])[0].strip()
            content = form.get("content", [""])[0].strip()
            if not title:
                self.send_html(render_edit_form(notes[note_id], error="Title is required."))
                return
            notes[note_id]["title"] = title
            notes[note_id]["content"] = content
            notes[note_id]["updated_at"] = datetime.utcnow().isoformat()
            save_notes(notes)
            self.redirect("/")

        elif path.startswith("/delete/"):
            note_id = path[len("/delete/"):]
            notes = load_notes()
            if note_id in notes:
                del notes[note_id]
                save_notes(notes)
            self.redirect("/")

        else:
            self.send_html(render_404(), status=404)

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), NotesHandler)
    print(f"Notes app running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()