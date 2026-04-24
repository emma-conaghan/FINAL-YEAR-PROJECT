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
    session_id = f"sess_{session_counter}_{os.urandom(8).hex()}"
    sessions[session_id] = username
    return session_id

def get_session_user(cookie_header):
    if not cookie_header:
        return None
    cookie = http.cookies.SimpleCookie()
    cookie.load(cookie_header)
    if 'session_id' in cookie:
        session_id = cookie['session_id'].value
        return sessions.get(session_id)
    return None

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        path = self.path
        cookie_header = self.headers.get('Cookie', '')
        username = get_session_user(cookie_header)

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html><head><title>Company Portal</title>
<style>
body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; background: #f5f5f5; }
.container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #333; }
a { display: inline-block; margin: 10px 10px 10px 0; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
a:hover { background: #0056b3; }
</style></head><body>
<div class="container">
<h1>Welcome to Company Portal</h1>'''
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

        elif path == '/login':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html><head><title>Login - Company Portal</title>
<style>
body { font-family: Arial, sans-serif; max-width: 400px; margin: 50px auto; background: #f5f5f5; }
.container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #333; }
input { width: 100%; padding: 10px; margin: 5px 0 15px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:hover { background: #0056b3; }
a { color: #007bff; }
</style></head><body>
<div class="container">
<h1>Login</h1>
<form method="POST" action="/login">
<label>Username:</label><input type="text" name="username" required>
<label>Password:</label><input type="password" name="password" required>
<button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="/register">Register here</a></p>
<p><a href="/">Back to Home</a></p>
</div></body></html>'''
            self.wfile.write(html.encode())

        elif path == '/register':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html><head><title>Register - Company Portal</title>
<style>
body { font-family: Arial, sans-serif; max-width: 400px; margin: 50px auto; background: #f5f5f5; }
.container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #333; }
input { width: 100%; padding: 10px; margin: 5px 0 15px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
button { width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:hover { background: #218838; }
a { color: #007bff; }
</style></head><body>
<div class="container">
<h1>Register</h1>
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

        elif path == '/welcome':
            if not username:
                self.send_response(302)
                self.send_header('Location', '/login')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            conn.close()
            html = f'''<!DOCTYPE html>
<html><head><title>Welcome - Company Portal</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; background: #f5f5f5; }}
.container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
h1 {{ color: #333; }}
.welcome-box {{ background: #e8f5e9; padding: 20px; border-radius: 4px; margin: 20px 0; }}
a {{ display: inline-block; margin: 10px 10px 10px 0; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }}
a:hover {{ background: #0056b3; }}
.logout {{ background: #dc3545; }}
.logout:hover {{ background: #c82333; }}
</style></head><body>
<div class="container">
<h1>Welcome, {username}!</h1>
<div class="welcome-box">
<p>You are successfully logged into the Company Portal.</p>
<p>Your user ID is: {user["id"] if user else "N/A"}</p>
<p>Role: {"Administrator" if user and user["is_admin"] else "Regular User"}</p>
</div>'''
            if user and user['is_admin']:
                html += '<a href="/admin">Admin Panel</a>'
            html += '<a href="/">Home</a>'
            html += '<a href="/logout" class="logout">Logout</a>'
            html += '</div></body></html>'
            self.wfile.write(html.encode())

        elif path == '/admin':
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
a { color: #007bff; }</style></head><body>
<h1>403 - Access Denied</h1><p>You do not have admin privileges.</p>
<a href="/">Back to Home</a></body></html>'''
                self.wfile.write(html.encode())
                return

            all_users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
            conn.close()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html><head><title>Admin Panel - Company Portal</title>
<style>
body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; background: #f5f5f5; }
.container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #333; }
table { width: 100%; border-collapse: collapse; margin: 20px 0; }
th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
th { background: #007bff; color: white; }
tr:hover { background: #f5f5f5; }
.badge { padding: 3px 8px; border-radius: 3px; font-size: 12px; color: white; }
.admin-badge { background: #dc3545; }
.user-badge { background: #28a745; }
a { display: inline-block; margin: 10px 10px 10px 0; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
a:hover { background: #0056b3; }
</style></head><body>
<div class="container">
<h1>Admin Panel</h1>
<h2>All Registered Users</h2>
<table>
<tr><th>ID</th><th>Username</th><th>Role</th></tr>'''
            for u in all_users:
                role = '<span class="badge admin-badge">Admin</span>' if u['is_admin'] else '<span class="badge user-badge">User</span>'
                html += f'<tr><td>{u["id"]}</td><td>{u["username"]}</td><td>{role}</td></tr>'
            html += f'''</table>
<p>Total users: {len(all_users)}</p>
<a href="/welcome">Dashboard</a>
<a href="/">Home</a>
</div></body></html>'''
            self.wfile.write(html.encode())

        elif path == '/logout':
            cookie = http.cookies.SimpleCookie()
            if cookie_header:
                cookie.load(cookie_header)
                if 'session_id' in cookie:
                    session_id = cookie['session_id'].value
                    if session_id in sessions:
                        del sessions[session_id]
            self.send_response(302)
            expire_cookie = http.cookies.SimpleCookie()
            expire_cookie['session_id'] = ''
            expire_cookie['session_id']['expires'] = 'Thu, 01 Jan 1970 00:00:00 GMT'
            expire_cookie['session_id']['path'] = '/'
            self.send_header('Set-Cookie', expire_cookie['session_id'].OutputString())
            self.send_header('Location', '/')
            self.end_headers()

        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html><html><head><title>404</title>
<style>body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; }
a { color: #007bff; }</style></head><body>
<h1>404 - Page Not Found</h1><a href="/">Back to Home</a></body></html>'''
            self.wfile.write(html.encode())

    def do_POST(self):
        path = self.path
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()
        params = parse_qs(body)

        def get_param(name):
            values = params.get(name, [])
            return values[0] if values else ''

        if path == '/login':
            login_username = get_param('username')
            password = get_param('password')

            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                                (login_username, password)).fetchone()
            conn.close()

            if user:
                session_id = create_session(login_username)
                self.send_response(302)
                cookie = http.cookies.SimpleCookie()
                cookie['session_id'] = session_id
                cookie['session_id']['path'] = '/'
                self.send_header('Set-Cookie', cookie['session_id'].OutputString())
                self.send_header('Location', '/welcome')
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                html = '''<!DOCTYPE html>
<html><head><title>Login Failed - Company Portal</title>
<style>
body { font-family