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

class PortalHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        cookie_header = self.headers.get('Cookie', '')
        username = get_session_user(cookie_header)

        if self.path == '/':
            self.send_html(self.home_page())
        elif self.path == '/login':
            self.send_html(self.login_page())
        elif self.path == '/register':
            self.send_html(self.register_page())
        elif self.path == '/welcome':
            if username:
                self.send_html(self.welcome_page(username))
            else:
                self.redirect('/login')
        elif self.path == '/admin':
            if username:
                conn = get_db()
                user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
                conn.close()
                if user and user['is_admin']:
                    self.send_html(self.admin_page())
                else:
                    self.send_html(self.error_page("Access Denied", "You do not have admin privileges."))
            else:
                self.redirect('/login')
        elif self.path == '/logout':
            cookie = http.cookies.SimpleCookie()
            cookie.load(cookie_header)
            if 'session_id' in cookie:
                session_id = cookie['session_id'].value
                sessions.pop(session_id, None)
            self.send_response(302)
            self.send_header('Location', '/')
            self.send_header('Set-Cookie', 'session_id=; Max-Age=0; Path=/')
            self.end_headers()
        else:
            self.send_html(self.error_page("404 Not Found", "Page not found."), code=404)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        params = parse_qs(post_data)

        if self.path == '/login':
            username = params.get('username', [''])[0].strip()
            password = params.get('password', [''])[0].strip()
            if not username or not password:
                self.send_html(self.login_page(error="Please fill in all fields."))
                return
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                                (username, password)).fetchone()
            conn.close()
            if user:
                session_id = create_session(username)
                self.send_response(302)
                self.send_header('Location', '/welcome')
                self.send_header('Set-Cookie', f'session_id={session_id}; Path=/')
                self.end_headers()
            else:
                self.send_html(self.login_page(error="Invalid username or password."))

        elif self.path == '/register':
            username = params.get('username', [''])[0].strip()
            password = params.get('password', [''])[0].strip()
            confirm = params.get('confirm_password', [''])[0].strip()
            if not username or not password or not confirm:
                self.send_html(self.register_page(error="Please fill in all fields."))
                return
            if password != confirm:
                self.send_html(self.register_page(error="Passwords do not match."))
                return
            if len(username) < 3:
                self.send_html(self.register_page(error="Username must be at least 3 characters."))
                return
            if len(password) < 4:
                self.send_html(self.register_page(error="Password must be at least 4 characters."))
                return
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                conn.close()
                session_id = create_session(username)
                self.send_response(302)
                self.send_header('Location', '/welcome')
                self.send_header('Set-Cookie', f'session_id={session_id}; Path=/')
                self.end_headers()
            except sqlite3.IntegrityError:
                conn.close()
                self.send_html(self.register_page(error="Username already exists."))
        else:
            self.send_html(self.error_page("404", "Not found"), code=404)

    def send_html(self, html, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def redirect(self, location):
        self.send_response(302)
        self.send_header('Location', location)
        self.end_headers()

    def base_template(self, title, body, username=None):
        nav_links = ""
        if username:
            nav_links = f"""
                <a href="/welcome">Welcome</a>
                <a href="/admin">Admin</a>
                <a href="/logout">Logout ({self.escape(username)})</a>
            """
        else:
            nav_links = """
                <a href="/">Home</a>
                <a href="/login">Login</a>
                <a href="/register">Register</a>
            """
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.escape(title)} - Company Portal</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; color: #333; }}
        nav {{ background: #2c3e50; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }}
        nav .brand {{ color: white; font-size: 1.3em; font-weight: bold; }}
        nav .links a {{ color: #ecf0f1; text-decoration: none; margin-left: 20px; font-size: 0.95em; }}
        nav .links a:hover {{ color: #3498db; }}
        .container {{ max-width: 600px; margin: 60px auto; padding: 0 20px; }}
        .card {{ background: white; border-radius: 10px; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ margin-bottom: 20px; color: #2c3e50; }}
        h2 {{ margin-bottom: 15px; color: #2c3e50; }}
        .form-group {{ margin-bottom: 20px; }}
        label {{ display: block; margin-bottom: 5px; font-weight: 600; color: #555; }}
        input[type="text"], input[type="password"] {{
            width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px;
            font-size: 1em; transition: border-color 0.3s;
        }}
        input[type="text"]:focus, input[type="password"]:focus {{ border-color: #3498db; outline: none; }}
        button {{ background: #3498db; color: white; border: none; padding: 12px 30px;
                 border-radius: 6px; font-size: 1em; cursor: pointer; width: 100%; }}
        button:hover {{ background: #2980b9; }}
        .error {{ background: #e74c3c; color: white; padding: 10px 15px; border-radius: 6px; margin-bottom: 20px; }}
        .success {{ background: #27ae60; color: white; padding: 10px 15px; border-radius: 6px; margin-bottom: 20px; }}
        .link {{ text-align: center; margin-top: 20px; }}
        .link a {{ color: #3498db; text-decoration: none; }}
        .link a:hover {{ text-decoration: underline; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #2c3e50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }}
        .badge-admin {{ background: #e74c3c; color: white; }}
        .badge-user {{ background: #3498db; color: white; }}
    </style>
</head>
<body>
    <nav>
        <div class="brand">🏢 Company Portal</div>
        <div class="links">{nav_links}</div>
    </nav>
    <div class="container">
        {body}
    </div>
</body>
</html>"""

    def escape(self, text):
        if text is None:
            return ""
        return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')

    def home_page(self):
        body = """
        <div class="card">
            <h1>Welcome to the Company Portal</h1>
            <p style="margin-bottom: 20px; color: #666; line-height: 1.6;">
                This is the internal company portal. Please log in or register to access the system.
            </p>
            <div style="display: flex; gap: 15px;">
                <a href="/login" style="flex:1;"><button type="button">Login</button></a>
                <a href="/register" style="flex:1;"><button type="button" style="background:#27ae60;">Register</button></a>
            </div>
        </div>
        """
        return self.base_template("Home", body)

    def login_page(self, error=None):
        error_html = f'<div class="error">{self.escape(error)}</div>' if error else ''
        body = f"""
        <div class="card">
            <h2>Login</h2>
            {error_html}
            <form method="POST" action="/login">
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit">Login</button>
            </form>
            <div class="link">
                <p style="margin-top: 15px;">Don't have an account? <a href="/register">Register here</a></p>
            </div>
        </div>
        """
        return self.base_template("Login", body)

    def register_page(self, error=None):
        error_html = f'<div class="error">{self.escape(error)}</div>' if error else ''
        body = f"""
        <div class="card">
            <h2>Register</h2>
            {error_html}
            <form method="POST" action="/register">
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required minlength="3">
                </div>
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required minlength="4">
                </div>
                <div class="form-group">
                    <label for="confirm_password">Confirm Password</label>
                    <input type="password" id="confirm_password" name="confirm_password" required>
                </div>
                <button type="submit" style="background: #27ae60;">Register</button>
            </form>
            <div class="link">
                <p style="margin-top: 15px;">Already have an account? <a href="/login">Login here</a></p>
            </div>
        </div>
        """
        return self.base_template("Register", body)

    def welcome_page(self, username):
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        role = "Administrator" if user and user['is_admin'] else "User"
        admin_link = ""
        if user and user['is_admin']:
            admin_link = '<p style="margin-top: 15px;"><a href="/admin"><button type="button" style="background:#e74c3c;">Go to Admin Panel</button></a></p>'
        body = f"""
        <div class="card">
            <h1>Welcome, {self.escape(username)}! 👋</h1>
            <p style="color: #666; margin-bottom: 10px; line-height: 1.6;">
                You are successfully logged in to the Company Portal.
            </p>
            <p style="margin-bottom: 15px;">
                <strong>Role:</strong> <span class="badge {'badge-admin' if role == 'Administrator' else 'badge-user'}">{role}</span>
            </p>
            <p style="color: #888;">
                This is your dashboard. From here you can access internal company resources.
            </p>
            {admin_link}
        </div>
        """
        return self.base_template("Welcome", body, username=username)

    def admin_page(self):
        cookie_header