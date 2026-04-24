import json
import os
import uuid
from datetime import datetime
import anthropic
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

client = anthropic.Anthropic()

NOTES_FILE = "notes.json"

def load_notes():
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_notes(notes):
    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)

def enhance_note_with_ai(title, content):
    enhanced_content = ""
    
    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Please enhance and improve this note while keeping its core meaning. Add helpful structure if appropriate.\n\nTitle: {title}\n\nContent: {content}\n\nProvide the enhanced content only, no explanations."
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            enhanced_content += text
    
    return enhanced_content

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Personal Notes App</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; color: #2c3e50; margin-bottom: 30px; font-size: 2.5em; }}
        .form-section {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .form-section h2 {{ color: #3498db; margin-bottom: 20px; }}
        input[type="text"], textarea {{ width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 16px; transition: border-color 0.3s; }}
        input[type="text"]:focus, textarea:focus {{ border-color: #3498db; outline: none; }}
        textarea {{ height: 150px; resize: vertical; margin-top: 10px; }}
        .btn {{ padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; margin: 5px; transition: transform 0.2s, box-shadow 0.2s; }}
        .btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
        .btn-primary {{ background: #3498db; color: white; }}
        .btn-ai {{ background: #9b59b6; color: white; }}
        .btn-edit {{ background: #f39c12; color: white; padding: 8px 16px; font-size: 14px; }}
        .btn-delete {{ background: #e74c3c; color: white; padding: 8px 16px; font-size: 14px; }}
        .notes-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .note-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: transform 0.2s; }}
        .note-card:hover {{ transform: translateY(-3px); }}
        .note-title {{ font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
        .note-content {{ color: #666; line-height: 1.6; margin-bottom: 15px; white-space: pre-wrap; }}
        .note-date {{ font-size: 0.8em; color: #999; margin-bottom: 10px; }}
        .note-actions {{ display: flex; justify-content: flex-end; gap: 5px; }}
        .loading {{ display: none; text-align: center; color: #9b59b6; margin: 10px 0; }}
        .label {{ font-weight: bold; color: #555; margin-bottom: 5px; display: block; }}
        .hidden-input {{ display: none; }}
        #editModal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; justify-content: center; align-items: center; }}
        #editModal.active {{ display: flex; }}
        .modal-content {{ background: white; padding: 30px; border-radius: 12px; width: 90%; max-width: 600px; }}
        .modal-content h2 {{ color: #3498db; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 Personal Notes</h1>
        
        <div class="form-section">
            <h2>Create New Note</h2>
            <form id="noteForm" method="POST" action="/create">
                <label class="label" for="title">Title:</label>
                <input type="text" id="title" name="title" placeholder="Enter note title..." required>
                <label class="label" for="content" style="margin-top: 15px;">Content:</label>
                <textarea id="content" name="content" placeholder="Write your note here..."></textarea>
                <div style="margin-top: 15px;">
                    <button type="submit" class="btn btn-primary">💾 Save Note</button>
                    <button type="button" class="btn btn-ai" onclick="enhanceNote()">✨ Enhance with AI</button>
                </div>
                <div class="loading" id="loading">🤖 AI is enhancing your note...</div>
            </form>
        </div>
        
        <h2 style="color: #2c3e50; margin-bottom: 20px;">Your Notes ({count})</h2>
        <div class="notes-grid">
            {notes_html}
        </div>
    </div>
    
    <div id="editModal">
        <div class="modal-content">
            <h2>Edit Note</h2>
            <form method="POST" action="/edit">
                <input type="hidden" id="editId" name="note_id">
                <label class="label" for="editTitle">Title:</label>
                <input type="text" id="editTitle" name="title" required>
                <label class="label" for="editContent" style="margin-top: 15px;">Content:</label>
                <textarea id="editContent" name="content"></textarea>
                <div style="margin-top: 15px;">
                    <button type="submit" class="btn btn-primary">💾 Save Changes</button>
                    <button type="button" class="btn btn-delete" onclick="closeModal()">Cancel</button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        function enhanceNote() {{
            const title = document.getElementById('title').value;
            const content = document.getElementById('content').value;
            
            if (!title && !content) {{
                alert('Please enter a title or content first!');
                return;
            }}
            
            document.getElementById('loading').style.display = 'block';
            
            fetch('/enhance', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
                body: `title=${{encodeURIComponent(title)}}&content=${{encodeURIComponent(content)}}`
            }})
            .then(r => r.json())
            .then(data => {{
                document.getElementById('content').value = data.enhanced_content;
                document.getElementById('loading').style.display = 'none';
            }})
            .catch(err => {{
                console.error(err);
                document.getElementById('loading').style.display = 'none';
                alert('Error enhancing note. Please try again.');
            }});
        }}
        
        function editNote(id, title, content) {{
            document.getElementById('editId').value = id;
            document.getElementById('editTitle').value = title;
            document.getElementById('editContent').value = content;
            document.getElementById('editModal').classList.add('active');
        }}
        
        function closeModal() {{
            document.getElementById('editModal').classList.remove('active');
        }}
        
        function deleteNote(id) {{
            if (confirm('Are you sure you want to delete this note?')) {{
                fetch('/delete', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
                    body: `note_id=${{id}}`
                }})
                .then(() => window.location.reload());
            }}
        }}
    </script>
</body>
</html>"""

def generate_note_card(note_id, note):
    content_escaped = note['content'].replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
    title_escaped = note['title'].replace("'", "\\'").replace('"', '\\"')
    
    return f"""
    <div class="note-card">
        <div class="note-title">{note['title']}</div>
        <div class="note-date">📅 {note.get('created_at', 'Unknown date')}</div>
        <div class="note-content">{note['content']}</div>
        <div class="note-actions">
            <button class="btn btn-edit" onclick="editNote('{note_id}', '{title_escaped}', '{content_escaped}')">✏️ Edit</button>
            <button class="btn btn-delete" onclick="deleteNote('{note_id}')">🗑️ Delete</button>
        </div>
    </div>"""

class NotesHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
    
    def send_html(self, html, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def get_post_data(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode()
        return parse_qs(post_data)
    
    def do_GET(self):
        if self.path == '/':
            notes = load_notes()
            notes_html = ''.join([generate_note_card(nid, note) for nid, note in notes.items()])
            if not notes_html:
                notes_html = '<p style="color: #999; text-align: center; grid-column: 1/-1;">No notes yet. Create your first note above!</p>'
            
            html = HTML_TEMPLATE.format(
                notes_html=notes_html,
                count=len(notes)
            )
            self.send_html(html)
        else:
            self.send_html("<h1>404 Not Found</h1>", 404)
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/create':
            data = self.get_post_data()
            title = data.get('title', [''])[0]
            content = data.get('content', [''])[0]
            
            if title:
                notes = load_notes()
                note_id = str(uuid.uuid4())
                notes[note_id] = {
                    'title': title,
                    'content': content,
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
                save_notes(notes)
            
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
        
        elif parsed_path.path == '/edit':
            data = self.get_post_data()
            note_id = data.get('note_id', [''])[0]
            title = data.get('title', [''])[0]
            content = data.get('content', [''])[0]
            
            notes = load_notes()
            if note_id in notes:
                notes[note_id]['title'] = title
                notes[note_id]['content'] = content
                notes[note_id]['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                save_notes(notes)
            
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
        
        elif parsed_path.path == '/delete':
            data = self.get_post_data()
            note_id = data.get('note_id', [''])[0]
            
            notes = load_notes()
            if note_id in notes:
                del notes[note_id]
                save_notes(notes)
            
            self.send_json({'success': True})
        
        elif parsed_path.path == '/enhance':
            data = self.get_post_data()
            title = data.get('title', [''])[0]
            content = data.get('content', [''])[0]
            
            try:
                enhanced = enhance_note_with_ai(title, content)
                self.send_json({'enhanced_content': enhanced})
            except Exception as e:
                self.send_json({'error': str(e)}, 500)
        
        else:
            self.send_html("<h1>404 Not Found</h1>", 404)

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8080), NotesHandler)
    print("Notes app running at http://localhost:8080")
    server.serve_forever()