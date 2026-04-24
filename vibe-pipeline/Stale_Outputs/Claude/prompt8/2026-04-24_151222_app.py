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
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personal Notes</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: Arial, sans-serif; background: #f0f2f5; color: #333; }}
        .container {{ max-width: 900px; margin: 30px auto; padding: 0 20px; }}
        h1 {{ text-align: center; margin-bottom: 30px; color: #2c3e50; font-size: 2em; }}
        h2 {{ color: #2c3e50; margin-bottom: 15px; }}
        .form-section {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        input[type=text], textarea {{ width: 100%; padding: 10px; margin-bottom: 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; }}
        textarea {{ height: 120px; resize: vertical; }}
        button {{ padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; }}
        .btn-primary {{ background: #3498db; color: white; }}
        .btn-primary:hover {{ background: #2980b9; }}
        .btn-danger {{ background: #e74c3c; color: white; font-size: 0.85em; padding: 6px 12px; }}
        .btn-danger:hover {{ background: #c0392b; }}
        .btn-edit {{ background: #2ecc71; color: white; font-size: 0.85em; padding: 6px 12px; }}
        .btn-edit:hover {{ background: #27ae60; }}
        .notes-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }}
        .note-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); position: relative; }}
        .note-title {{ font-size: 1.1em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; word-break: break-word; }}
        .note-content {{ color: #555; white-space: pre-wrap; word-break: break-word; margin-bottom: 15px; max-height: 150px; overflow-y: auto; }}
        .note-actions {{ display: flex; gap: 8px; }}
        .note-meta {{ font-size: 0.75em; color: #999; margin-bottom: 10px; }}
        .no-notes {{ text-align: center; color: #999; font-size: 1.1em; padding: 40px; }}
        .message {{ padding: 12px; border-radius: 4px; margin-bottom: 20px; text-align: center; }}
        .success {{ background: #d4edda; color: #155724; }}
        .error {{ background: #f8d7da; color: #721c24; }}
        .edit-form {{ background: #fffde7; border: 1px solid #f9a825; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📝 Personal Notes</h1>
    {message}
    <div class="form-section {edit_class}">
        <h2>{form_title}</h2>
        <form method="POST" action="{form_action}">
            {hidden_id}
            <input type="text" name="title" placeholder="Note title..." value="{edit_title}" required maxlength="200">
            <textarea name="content" placeholder="Write your note here...">{edit_content}</textarea>
            <button type="submit" class="btn-primary">{submit_label}</button>
            {cancel_button}
        </form>
    </div>
    <div>
        <h2>Your Notes ({count})</h2>
        <br>
        {notes_html}
    </div>
</div>
</body>
</html>"""

NOTE_CARD_TEMPLATE = """<div class="note-card">
    <div class="note-title">{title}</div>
    <div class="note-meta">ID: {note_id}</div>
    <div class="note-content">{content}</div>
    <div class="note-actions">
        <a href="/edit?id={note_id}"><button class="btn-edit">✏️ Edit</button></a>
        <form method="POST" action="/delete" style="display:inline;">
            <input type="hidden" name="id" value="{note_id}">
            <button type="submit" class="btn-danger" onclick="return confirm('Delete this note?')">🗑️ Delete</button>
        </form>
    </div>
</div>"""

class NotesHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def send_html(self, content, status=200):
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def parse_body(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        return parse_qs(body, keep_blank_values=True)

    def render_page(self, message="", edit_note=None):
        notes = load_notes()

        if message.startswith("SUCCESS:"):
            msg_html = f'<div class="message success">{html.escape(message[8:])}</div>'
        elif message.startswith("ERROR:"):
            msg_html = f'<div class="message error">{html.escape(message[6:])}</div>'
        else:
            msg_html = ""

        if edit_note:
            form_title = "Edit Note"
            form_action = "/update"
            hidden_id = f'<input type="hidden" name="id" value="{edit_note["id"]}">'
            edit_title = html.escape(edit_note["title"])
            edit_content = html.escape(edit_note["content"])
            submit_label = "💾 Save Changes"
            cancel_button = '<a href="/"><button type="button" style="margin-left:8px;background:#95a5a6;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;font-size:1em;">Cancel</button></a>'
            edit_class = "edit-form"
        else:
            form_title = "Add New Note"
            form_action = "/create"
            hidden_id = ""
            edit_title = ""
            edit_content = ""
            submit_label = "➕ Add Note"
            cancel_button = ""
            edit_class = ""

        if notes:
            cards = []
            for note in reversed(notes):
                card = NOTE_CARD_TEMPLATE.format(
                    title=html.escape(note["title"]),
                    note_id=note["id"],
                    content=html.escape(note["content"])
                )
                cards.append(card)
            notes_html = '<div class="notes-grid">' + "".join(cards) + "</div>"
        else:
            notes_html = '<div class="no-notes">No notes yet. Create your first note above!</div>'

        page = HTML_TEMPLATE.format(
            message=msg_html,
            form_title=form_title,
            form_action=form_action,
            hidden_id=hidden_id,
            edit_title=edit_title,
            edit_content=edit_content,
            submit_label=submit_label,
            cancel_button=cancel_button,
            edit_class=edit_class,
            count=len(notes),
            notes_html=notes_html
        )
        return page

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "":
            page = self.render_page()
            self.send_html(page)

        elif path == "/edit":
            note_id = query.get("id", [None])[0]
            if note_id is None:
                self.redirect("/")
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.redirect("/")
                return
            notes = load_notes()
            note = next((n for n in notes if n["id"] == note_id), None)
            if note is None:
                page = self.render_page(message="ERROR:Note not found.")
                self.send_html(page)
                return
            page = self.render_page(edit_note=note)
            self.send_html(page)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        data = self.parse_body()

        if path == "/create":
            title = data.get("title", [""])[0].strip()
            content = data.get("content", [""])[0].strip()
            if not title:
                page = self.render_page(message="ERROR:Title cannot be empty.")
                self.send_html(page)
                return
            notes = load_notes()
            new_note = {
                "id": get_next_id(notes),
                "title": title,
                "content": content
            }
            notes.append(new_note)
            save_notes(notes)
            page = self.render_page(message="SUCCESS:Note created successfully!")
            self.send_html(page)

        elif path == "/update":
            note_id = data.get("id", [None])[0]
            title = data.get("title", [""])[0].strip()
            content = data.get("content", [""])[0].strip()
            if not note_id:
                self.redirect("/")
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.redirect("/")
                return
            if not title:
                notes = load_notes()
                note = next((n for n in notes if n["id"] == note_id), None)
                page = self.render_page(message="ERROR:Title cannot be empty.", edit_note=note)
                self.send_html(page)
                return
            notes = load_notes()
            updated = False
            for note in notes:
                if note["id"] == note_id:
                    note["title"] = title
                    note["content"] = content
                    updated = True
                    break
            if updated:
                save_notes(notes)
                page = self.render_page(message="SUCCESS:Note updated successfully!")
            else:
                page = self.render_page(message="ERROR:Note not found.")
            self.send_html(page)

        elif path == "/delete":
            note_id = data.get("id", [None])[0]
            if not note_id:
                self.redirect("/")
                return
            try:
                note_id = int(note_id)
            except ValueError:
                self.redirect("/")
                return
            notes = load_notes()
            new_notes = [n for n in notes if n["id"] != note_id]
            if len(new_notes) < len(notes):
                save_notes(new_notes)
                page = self.render_page(message="SUCCESS:Note deleted successfully!")
            else:
                page = self.render_page(message="ERROR:Note not found.")
            self.send_html(page)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")


def main():
    host = "0.0.0.0"
    port = 8080
    server = HTTPServer((host, port), NotesHandler)
    print(f"Notes app running at http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()