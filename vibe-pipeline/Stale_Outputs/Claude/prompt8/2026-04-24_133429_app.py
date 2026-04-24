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
        json.dump(notes, f, indent=2)

def get_next_id(notes):
    if not notes:
        return 1
    return max(n["id"] for n in notes) + 1

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Personal Notes</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .nav {{ margin-bottom: 20px; }}
  .nav a {{ margin-right: 10px; background: #007bff; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px; }}
  .nav a:hover {{ background: #0056b3; }}
  .note-card {{ background: white; border-radius: 6px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .note-card h2 {{ margin-top: 0; color: #444; }}
  .note-card p {{ white-space: pre-wrap; color: #555; }}
  .note-actions a {{ margin-right: 8px; color: #007bff; text-decoration: none; }}
  .note-actions a.delete {{ color: #dc3545; }}
  form {{ background: white; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  input[type=text], textarea {{ width: 100%; padding: 8px; margin-bottom: 12px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }}
  textarea {{ height: 200px; resize: vertical; }}
  input[type=submit] {{ background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
  input[type=submit]:hover {{ background: #1e7e34; }}
  .empty {{ color: #888; font-style: italic; }}
</style>
</head>
<body>
<h1>📝 Personal Notes</h1>
<div class="nav">
  <a href="/">All Notes</a>
  <a href="/new">New Note</a>
</div>
{content}
</body>
</html>"""

def render_notes_list(notes):
    if not notes:
        content = '<p class="empty">No notes yet. <a href="/new">Create your first note!</a></p>'
    else:
        cards = []
        for note in reversed(notes):
            title = html.escape(note["title"])
            body = html.escape(note["content"])
            nid = note["id"]
            card = f"""<div class="note-card">
<h2>{title}</h2>
<p>{body}</p>
<div class="note-actions">
  <a href="/edit?id={nid}">✏️ Edit</a>
  <a href="/delete?id={nid}" class="delete" onclick="return confirm('Delete this note?')">🗑️ Delete</a>
</div>
</div>"""
            cards.append(card)
        content = "\n".join(cards)
    return HTML_TEMPLATE.format(content=content)

def render_new_form():
    content = """<h2>New Note</h2>
<form method="POST" action="/create">
  <label>Title:</label><br>
  <input type="text" name="title" placeholder="Note title" required><br>
  <label>Content:</label><br>
  <textarea name="content" placeholder="Write your note here..."></textarea><br>
  <input type="submit" value="Save Note">
</form>"""
    return HTML_TEMPLATE.format(content=content)

def render_edit_form(note):
    title = html.escape(note["title"])
    content_val = html.escape(note["content"])
    nid = note["id"]
    content = f"""<h2>Edit Note</h2>
<form method="POST" action="/update">
  <input type="hidden" name="id" value="{nid}">
  <label>Title:</label><br>
  <input type="text" name="title" value="{title}" required><br>
  <label>Content:</label><br>
  <textarea name="content">{content_val}</textarea><br>
  <input type="submit" value="Update Note">
</form>"""
    return HTML_TEMPLATE.format(content=content)

def render_error(message):
    content = f'<p style="color:red;">{html.escape(message)}</p><a href="/">Go back</a>'
    return HTML_TEMPLATE.format(content=content)

class NotesHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

    def send_html(self, body, status=200):
        encoded = body.encode("utf-8")
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

        if path == "/":
            notes = load_notes()
            self.send_html(render_notes_list(notes))

        elif path == "/new":
            self.send_html(render_new_form())

        elif path == "/edit":
            note_id = params.get("id", [None])[0]
            if note_id is None:
                self.send_html(render_error("No note ID provided."), 400)
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.send_html(render_error("Invalid note ID."), 400)
                return
            notes = load_notes()
            note = next((n for n in notes if n["id"] == note_id), None)
            if note is None:
                self.send_html(render_error("Note not found."), 404)
                return
            self.send_html(render_edit_form(note))

        elif path == "/delete":
            note_id = params.get("id", [None])[0]
            if note_id is None:
                self.send_html(render_error("No note ID provided."), 400)
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.send_html(render_error("Invalid note ID."), 400)
                return
            notes = load_notes()
            notes = [n for n in notes if n["id"] != note_id]
            save_notes(notes)
            self.redirect("/")

        else:
            self.send_html(render_error("Page not found."), 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/create":
            body = self.read_body()
            fields = parse_qs(body)
            title = fields.get("title", [""])[0].strip()
            content = fields.get("content", [""])[0].strip()
            if not title:
                self.send_html(render_error("Title is required."), 400)
                return
            notes = load_notes()
            note = {
                "id": get_next_id(notes),
                "title": title,
                "content": content
            }
            notes.append(note)
            save_notes(notes)
            self.redirect("/")

        elif path == "/update":
            body = self.read_body()
            fields = parse_qs(body)
            note_id = fields.get("id", [None])[0]
            title = fields.get("title", [""])[0].strip()
            content = fields.get("content", [""])[0].strip()
            if note_id is None:
                self.send_html(render_error("No note ID provided."), 400)
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.send_html(render_error("Invalid note ID."), 400)
                return
            if not title:
                self.send_html(render_error("Title is required."), 400)
                return
            notes = load_notes()
            found = False
            for note in notes:
                if note["id"] == note_id:
                    note["title"] = title
                    note["content"] = content
                    found = True
                    break
            if not found:
                self.send_html(render_error("Note not found."), 404)
                return
            save_notes(notes)
            self.redirect("/")

        else:
            self.send_html(render_error("Page not found."), 404)

def run():
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), NotesHandler)
    print(f"Notes app running at http://{host}:{port}")
    server.serve_forever()

if __name__ == "__main__":
    run()