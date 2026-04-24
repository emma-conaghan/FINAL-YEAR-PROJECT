import sqlite3
import os
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DATABASE = 'company_portal.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    cursor = conn.execute("SELECT * FROM users WHERE username = 'admin'")
    if cursor.fetchone() is None:
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     ('admin', 'admin123', 1))
    conn.commit()
    conn.close()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal - {{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; min-height: 100vh; }
        .navbar { background: #1a73e8; padding: 15px 30px; color: white; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .navbar a { color: white; text-decoration: none; margin-left: 20px; font-weight: 500; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 60px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #202124; margin-bottom: 20px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 6px; font-weight: 600; color: #5f6368; }
        input[type="text"], input[type="password"] { width: 100%; padding: 12px; border: 1px solid #dadce0; border-radius: 4px; font-size: 16px; }
        input[type="text"]:focus, input[type="password"]:focus { outline: none; border-color: #1a73e8; box-shadow: 0 0 0 2px rgba(26,115,232,0.2); }
        .btn { background: #1a73e8; color: white; padding: 12px 24px; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; width: 100%; font-weight: 600; }
        .btn:hover { background: #1557b0; }
        .btn-danger { background: #d93025; }
        .btn-danger:hover { background: #b3261e; }
        .message { padding: 12px; border-radius: 4px; margin-bottom: 20px; }
        .error { background: #fce8e6; color: #d93025; border: 1px solid #f5c6cb; }
        .success { background: #e6f4ea; color: #137333; border: 1px solid #c3e6cb; }
        .link { text-align: center; margin-top: 20px; }
        .link a { color: #1a73e8; text-decoration: none; }
        .link a:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #dadce0; }
        th { background: #f8f9fa; font-weight: 600; color: #5f6368; }
        tr:hover { background: #f8f9fa; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge-admin { background: #e8f0fe; color: #1a73e8; }
        .badge-user { background: #e6f4ea; color: #137333; }
        .welcome-header { font-size: 28px; }
        .wide-container { max-width: 900px; margin: 60px auto; padding: 0 20px; }
    </style>
</head>
<body>
    <div class="navbar">
        <strong>🏢 Company Portal</strong>
        <div>
            {% if session.get('username') %}
                <span>Welcome, {{ session['username'] }}</span>
                {% if session.get('is_admin') %}
                    <a href="/admin">Admin Panel</a>
                {% endif %}
                <a href="/dashboard">Dashboard</a>
                <a href="/logout">Logout</a>
            {% else %}
                <a href="/login">Login</a>
                <a href="/register">Register</a>
            {% endif %}
        </div>
    </div>
    {{ content }}
</body>
</html>
'''

LOGIN_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="container">
    <div class="card">
        <h2>Login to Your Account</h2>
        {% if error %}
            <div class="message error">{{ error }}</div>
        {% endif %}
        {% if success %}
            <div class="message success">{{ success }}</div>
        {% endif %}
        <form method="POST" action="/login">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required autofocus>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required>
            </div>
            <button type="submit" class="btn">Sign In</button>
        </form>
        <div class="link">
            <p>Don't have an account? <a href="/register">Register here</a></p>
        </div>
    </div>
</div>
{% endset %}
'''

REGISTER_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="container">
    <div class="card">
        <h2>Create an Account</h2>
        {% if error %}
            <div class="message error">{{ error }}</div>
        {% endif %}
        <form method="POST" action="/register">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required autofocus>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required>
            </div>
            <div class="form-group">
                <label>Confirm Password</label>
                <input type="password" name="confirm_password" required>
            </div>
            <button type="submit" class="btn">Create Account</button>
        </form>
        <div class="link">
            <p>Already have an account? <a href="/login">Login here</a></p>
        </div>
    </div>
</div>
{% endset %}
'''

DASHBOARD_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="container">
    <div class="card">
        <h1 class="welcome-header">👋 Welcome, {{ session['username'] }}!</h1>
        <p style="color: #5f6368; margin-top: 10px; font-size: 18px;">You are logged into the Company Portal.</p>
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #dadce0;">
        <h3 style="margin-bottom: 15px;">Quick Links</h3>
        <ul style="list-style: none; padding: 0;">
            <li style="padding: 10px 0; border-bottom: 1px solid #f0f0f0;">📧 Company Email</li>
            <li style="padding: 10px 0; border-bottom: 1px solid #f0f0f0;">📅 Calendar</li>
            <li style="padding: 10px 0; border-bottom: 1px solid #f0f0f0;">📋 Project Board</li>
            <li style="padding: 10px 0;">📚 Documentation</li>
        </ul>
        {% if session.get('is_admin') %}
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #dadce0;">
        <a href="/admin" class="btn" style="display: block; text-align: center; text-decoration: none;">Go to Admin Panel</a>
        {% endif %}
    </div>
</div>
{% endset %}
'''

ADMIN_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="wide-container">
    <div class="card">
        <h2>🔧 Admin Panel - Registered Users</h2>
        <p style="color: #5f6368; margin-bottom: 10px;">Total users: {{ users|length }}</p>
        {% if message %}
            <div class="message success">{{ message }}</div>
        {% endif %}
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Username</th>
                    <th>Password</th>
                    <th>Role</th>
                    <th>Created At</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {% for user in users %}
                <tr>
                    <td>{{ user['id'] }}</td>
                    <td><strong>{{ user['username'] }}</strong></td>
                    <td>{{ user['password'] }}</td>
                    <td>
                        {% if user['is_admin'] %}
                            <span class="badge badge-admin">Admin</span>
                        {% else %}
                            <span class="badge badge-user">User</span>
                        {% endif %}
                    </td>
                    <td>{{ user['created_at'] }}</td>
                    <td>
                        {% if not user['is_admin'] %}
                        <a href="/admin/delete/{{ user['id'] }}" style="color: #d93025; text-decoration: none;" onclick="return confirm('Delete user {{ user['username'] }}?')">Delete</a>
                        {% else %}
                        <span style="color: #999;">-</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>
{% endset %}
'''

HOME_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="container" style="text-align: center; margin-top: 100px;">
    <div class="card">
        <h1>🏢 Welcome to the Company Portal</h1>
        <p style="color: #5f6368; margin: 20px 0; font-size: 18px;">Your one-stop internal hub for all company resources.</p>
        <div style="display: flex; gap: 10px; justify-content: center; margin-top: 30px;">
            <a href="/login" class="btn" style="text-decoration: none; width: auto; padding: 12px 40px;">Login</a>
            <a href="/register" class="btn" style="text-decoration: none; width: auto; padding: 12px 40px; background: #34a853;">Register</a>
        </div>
    </div>
</div>
{% endset %}
'''

def render_page(template_str, title="Portal", **kwargs):
    from jinja2 import Environment
    env = Environment()
    base_tmpl = env.from_string(BASE_TEMPLATE)
    base_rendered = base_tmpl.render(session=session, title=title)
    full_template = template_str.replace('{% extends "base" %}', '')
    full_template = BASE_TEMPLATE.replace('{{ content }}', '') 
    combined = BASE_TEMPLATE.replace('{{ content }}', '{{ content_block }}')
    content_env = Environment()
    content_tmpl = content_env.from_string(template_str)
    content_html = ''
    import re
    match = re.search(r'\{% set content %\}(.*?)\{% endset %\}', template_str, re.DOTALL)
    if match:
        content_html = match.group(1)
    final_template = BASE_TEMPLATE.replace('{{ content }}', content_html)
    return render_template_string(final_template, title=title, session=session, **kwargs)


@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_page(HOME_PAGE, title="Home")


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not password:
            error = 'Username and password are required.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login', registered='1'))
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose a different one.'
                conn.close()

    return render_page(REGISTER_PAGE, title="Register", error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    success = None

    if request.args.get('registered'):
        success = 'Account created successfully! Please log in.'

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                            (username, password)).fetchone()
        conn.close()

        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password.'

    return render_page(LOGIN_PAGE, title="Login", error=error, success=success)


@app.route('/dashboard')
def dashboard():