import sqlite3
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import http.cookies

sessions = {}
session_counter = 0

def get_db():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )''')
    cursor = conn.execute("SELECT * FROM users WHERE username = 'admin'")
    if cursor.fetchone() is None:
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     ('admin', 'admin123', 1))
    conn.commit()
    conn.close()

def create_session(username):
    global session_counter
    session_counter += 1
    session_id = f"session_{session_counter}_{os.urandom(8).hex()}"
    sessions[session_id] = username
    return session_id

def get_session_user(cookie_header):
    if not cookie_header:
        return None
    cookie = http.cookies.SimpleCookie()
    try:
        cookie.load(cookie_header)
    except Exception:
        return None
    if 'session_id' in cookie:
        session_id = cookie['session_id'].value
        return sessions.get(session_id)
    return None

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        cookie_header = self.headers.get('Cookie')
        username = get_session_user(cookie_header)

        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html><head><title>Company Portal</title>
<style>
body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; background: #f5f5f5; }
.container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #333; }
a { display: inline-block; margin: 10px 5px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
a:hover { background: #0056b3; }
</style></head><body>
<div class="container">
<h1>Welcome to the Company Portal</h1>'''
            if username:
                html += f'<p>Logged in as: <strong>{username}</strong></p>'
                html += '<a href="/welcome">Dashboard</a>'
                html += '<a href="/logout">Logout</a>'
                conn = get_db()
                user = conn.execute("SELECT is_admin FROM users WHERE username = ?", (username,)).fetchone()
                conn.close()
                if user and user['is_admin']:
                    html += '<a href="/admin">Admin Panel</a>'
            else:
                html += '<a href="/login">Login</a>'
                html += '<a href="/register">Register</a>'
            html += '</div></body></html>'
            self.wfile.write(html.encode())

        elif self.path == '/login':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html><head><title>Login - Company Portal</title>
<style>
body { font-family: Arial, sans-serif; max-width: 400px; margin: 50px auto; background: #f5f5f5; }
.container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h2 { color: #333; }
input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ddd; border-radius: 4px; }
button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:hover { background: #0056b3; }
a { color: #007bff; }
</style></head><body>
<div class="container">
<h2>Login</h2>
<form method="POST" action="/login">
<label>Username:</label><input type="text" name="username" required>
<label>Password:</label><input type="password" name="password" required>
<button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="/register">Register here</a></p>
<p><a href="/">Back to Home</a></p>
</div></body></html>'''
            self.wfile.write(html.encode())

        elif self.path == '/register':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html><head><title>Register - Company Portal</title>
<style>
body { font-family: Arial, sans-serif; max-width: 400px; margin: 50px auto; background: #f5f5f5; }
.container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h2 { color: #333; }
input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ddd; border-radius: 4px; }
button { width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:hover { background: #218838; }
a { color: #007bff; }
</style></head><body>
<div class="container">
<h2>Register</h2>
<form method="POST" action="/register">
<label>Username:</label><input type="text" name="username" required>
<label>Password:</label><input type="password" name="password" required>
<label>Confirm Password:</label><input type="password" name="confirm_password" required>
<button type="submit">Register</button>
</form>
<p>Already have an account? <a href="/login">Login here</a></p>
<p><a href="/">Back to Home</a></p>
</div></body></html>'''
            self.wfile.write(html.encode())

        elif self.path == '/welcome':
            if not username:
                self.send_response(302)
                self.send_header('Location', '/login')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            conn = get_db()
            user = conn.execute("SELECT is_admin FROM users WHERE username = ?", (username,)).fetchone()
            conn.close()
            admin_link = ''
            if user and user['is_admin']:
                admin_link = '<a href="/admin" style="display:inline-block;margin:10px 5px;padding:10px 20px;background:#ffc107;color:#333;text-decoration:none;border-radius:4px;">Admin Panel</a>'
            html = f'''<!DOCTYPE html>
<html><head><title>Welcome - Company Portal</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; background: #f5f5f5; }}
.container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
h1 {{ color: #333; }}
.welcome-msg {{ background: #d4edda; padding: 15px; border-radius: 4px; color: #155724; }}
a {{ display: inline-block; margin: 10px 5px; padding: 10px 20px; background: #dc3545; color: white; text-decoration: none; border-radius: 4px; }}
a:hover {{ background: #c82333; }}
</style></head><body>
<div class="container">
<h1>Welcome, {username}!</h1>
<div class="welcome-msg">
<p>You are successfully logged in to the Company Portal.</p>
<p>This is your personal dashboard.</p>
</div>
<br>
{admin_link}
<a href="/logout">Logout</a>
<a href="/" style="background:#007bff;">Home</a>
</div></body></html>'''
            self.wfile.write(html.encode())

        elif self.path == '/admin':
            if not username:
                self.send_response(302)
                self.send_header('Location', '/login')
                self.end_headers()
                return
            conn = get_db()
            user = conn.execute("SELECT is_admin FROM users WHERE username = ?", (username,)).fetchone()
            if not user or not user['is_admin']:
                conn.close()
                self.send_response(403)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                html = '''<!DOCTYPE html>
<html><head><title>Access Denied</title>
<style>body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; }
.container { background: #f8d7da; padding: 30px; border-radius: 8px; color: #721c24; }
a { color: #007bff; }</style></head><body>
<div class="container"><h2>Access Denied</h2><p>You do not have admin privileges.</p>
<a href="/welcome">Back to Dashboard</a></div></body></html>'''
                self.wfile.write(html.encode())
                return

            users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
            conn.close()

            rows = ''
            for u in users:
                role = 'Admin' if u['is_admin'] else 'User'
                rows += f"<tr><td>{u['id']}</td><td>{u['username']}</td><td>{role}</td></tr>"

            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = f'''<!DOCTYPE html>
<html><head><title>Admin Panel - Company Portal</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 50px auto; background: #f5f5f5; }}
.container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
h2 {{ color: #333; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #007bff; color: white; }}
tr:hover {{ background: #f1f1f1; }}
a {{ display: inline-block; margin: 10px 5px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }}
</style></head><body>
<div class="container">
<h2>Admin Panel - Registered Users</h2>
<table>
<tr><th>ID</th><th>Username</th><th>Role</th></tr>
{rows}
</table>
<p>Total users: {len(users)}</p>
<a href="/welcome">Dashboard</a>
<a href="/logout" style="background:#dc3545;">Logout</a>
</div></body></html>'''
            self.wfile.write(html.encode())

        elif self.path == '/logout':
            session_id_to_remove = None
            if cookie_header:
                cookie = http.cookies.SimpleCookie()
                try:
                    cookie.load(cookie_header)
                    if 'session_id' in cookie:
                        session_id_to_remove = cookie['session_id'].value
                except Exception:
                    pass
            if session_id_to_remove and session_id_to_remove in sessions:
                del sessions[session_id_to_remove]
            self.send_response(302)
            self.send_header('Set-Cookie', 'session_id=; Max-Age=0; Path=/')
            self.send_header('Location', '/')
            self.end_headers()

        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<h1>404 Not Found</h1><a href="/">Go Home</a>')

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()
        params = parse_qs(body)

        if self.path == '/login':
            username = params.get('username', [''])[0].strip()
            password = params.get('password', [''])[0]

            if not username or not password:
                self.send_error_page('Login Failed', 'Please provide both username and password.', '/login')
                return

            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                                (username, password)).fetchone()
            conn.close()

            if user:
                session_id = create_session(username)
                self.send_response(302)
                self.send_header('Set-Cookie', f'session_id={session_id}; Path=/')
                self.send_header('Location', '/welcome')
                self.end_headers()
            else:
                self.send_error_page('Login Failed', 'Invalid username or password.', '/login')

        elif self.path == '/register':
            username = params.get('username', [''])[0].strip()
            password = params.get('password', [''])[0]
            confirm_password = params.get('confirm_password', [''])[0]

            if not username or not password:
                self.send_error_page('Registration Failed', 'Please fill in all fields.', '/register')
                return

            if len(username) < 3:
                self.send_error_page('Registration Failed', 'Username must be at least 3 characters.', '/register')
                return

            if len(password) < 4:
                self.send_error_page('Registration Failed', 'Password must be at least 4 characters.', '/register')
                return

            if password != confirm_password:
                self.send_error_page('Registration Failed', 'Passwords do not match.', '/register')
                return

            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username,