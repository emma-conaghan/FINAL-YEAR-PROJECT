import sqlite3
import os
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DB_PATH = 'portal.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
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
        .navbar { background: #2c3e50; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 20px; font-size: 14px; }
        .navbar a:hover { text-decoration: underline; }
        .navbar .brand { color: white; font-size: 20px; font-weight: bold; }
        .container { max-width: 600px; margin: 60px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h2 { margin-bottom: 25px; color: #2c3e50; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 6px; color: #555; font-weight: 600; font-size: 14px; }
        .form-group input { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        .form-group input:focus { outline: none; border-color: #3498db; }
        .btn { display: inline-block; padding: 12px 30px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; text-decoration: none; }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .alert { padding: 12px 20px; border-radius: 5px; margin-bottom: 20px; font-size: 14px; }
        .alert-error { background: #fdecea; color: #c0392b; border: 1px solid #f5c6cb; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .links { margin-top: 20px; text-align: center; font-size: 14px; }
        .links a { color: #3498db; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; color: #555; font-size: 13px; text-transform: uppercase; }
        tr:hover { background: #f8f9fa; }
        .badge { padding: 3px 10px; border-radius: 12px; font-size: 12px; }
        .badge-admin { background: #e74c3c; color: white; }
        .badge-user { background: #3498db; color: white; }
        .welcome-header { text-align: center; }
        .welcome-header h1 { font-size: 32px; color: #2c3e50; margin-bottom: 10px; }
        .welcome-header p { color: #777; font-size: 16px; }
    </style>
</head>
<body>
    <div class="navbar">
        <span class="brand">🏢 Company Portal</span>
        <div>
            {% if session.get('user_id') %}
                <a href="{{ url_for('welcome') }}">Home</a>
                {% if session.get('is_admin') %}
                    <a href="{{ url_for('admin_panel') }}">Admin</a>
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout ({{ session.get('username') }})</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </div>
    <div class="container">
        {{ content }}
    </div>
</body>
</html>
'''

LOGIN_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="card">
    <h2>Sign In</h2>
    {% if error %}
        <div class="alert alert-error">{{ error }}</div>
    {% endif %}
    <form method="POST">
        <div class="form-group">
            <label>Username</label>
            <input type="text" name="username" required placeholder="Enter your username">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" name="password" required placeholder="Enter your password">
        </div>
        <button type="submit" class="btn">Sign In</button>
    </form>
    <div class="links">
        Don't have an account? <a href="{{ url_for('register') }}">Register here</a>
    </div>
</div>
{% endset %}
'''

REGISTER_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="card">
    <h2>Create Account</h2>
    {% if error %}
        <div class="alert alert-error">{{ error }}</div>
    {% endif %}
    {% if success %}
        <div class="alert alert-success">{{ success }}</div>
    {% endif %}
    <form method="POST">
        <div class="form-group">
            <label>Username</label>
            <input type="text" name="username" required placeholder="Choose a username">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" name="password" required placeholder="Choose a password">
        </div>
        <div class="form-group">
            <label>Confirm Password</label>
            <input type="password" name="confirm_password" required placeholder="Confirm your password">
        </div>
        <button type="submit" class="btn btn-success">Create Account</button>
    </form>
    <div class="links">
        Already have an account? <a href="{{ url_for('login') }}">Sign in here</a>
    </div>
</div>
{% endset %}
'''

WELCOME_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="card">
    <div class="welcome-header">
        <h1>Welcome, {{ username }}! 👋</h1>
        <p>You are successfully logged into the Company Portal.</p>
    </div>
    <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
    <h3 style="color: #2c3e50; margin-bottom: 15px;">Quick Links</h3>
    <p style="color: #777; margin-bottom: 10px;">
        {% if is_admin %}
            As an administrator, you can <a href="{{ url_for('admin_panel') }}">manage users</a> from the admin panel.
        {% else %}
            Contact your administrator if you need any assistance.
        {% endif %}
    </p>
</div>
{% endset %}
'''

ADMIN_PAGE = '''
{% extends "base" %}
{% set content %}
<div class="card">
    <h2>Admin Panel - Registered Users</h2>
    <p style="color: #777; margin-bottom: 10px;">Total users: {{ users|length }}</p>
    {% if delete_success %}
        <div class="alert alert-success">{{ delete_success }}</div>
    {% endif %}
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Password</th>
                <th>Role</th>
                <th>Created</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
            {% for user in users %}
            <tr>
                <td>{{ user.id }}</td>
                <td>{{ user.username }}</td>
                <td>{{ user.password }}</td>
                <td>
                    {% if user.is_admin %}
                        <span class="badge badge-admin">Admin</span>
                    {% else %}
                        <span class="badge badge-user">User</span>
                    {% endif %}
                </td>
                <td>{{ user.created_at }}</td>
                <td>
                    {% if not user.is_admin %}
                        <a href="{{ url_for('delete_user', user_id=user.id) }}" class="btn btn-danger" style="padding: 5px 15px; font-size: 12px;" onclick="return confirm('Delete this user?')">Delete</a>
                    {% else %}
                        -
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endset %}
'''

def render_page(template_str, title="Portal", **kwargs):
    from jinja2 import Environment
    env = Environment()
    base_tmpl = env.from_string(BASE_TEMPLATE)
    full_template = template_str.replace('{% extends "base" %}', '')
    full_template = full_template.replace('{% set content %}', '').replace('{% endset %}', '')
    content_html = render_template_string(full_template, **kwargs)
    return render_template_string(BASE_TEMPLATE, title=title, content=render_template_string(full_template, **kwargs), session=session)


@app.route('/')
def index():
    if session.get('user_id'):
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('welcome'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        conn = get_db()
        query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
        try:
            user = conn.execute(query).fetchone()
        except Exception:
            user = None
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'

    content = render_template_string('''
    <div class="card">
        <h2>Sign In</h2>
        {% if error %}
            <div class="alert alert-error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required placeholder="Enter your username">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required placeholder="Enter your password">
            </div>
            <button type="submit" class="btn">Sign In</button>
        </form>
        <div class="links">
            Don't have an account? <a href="{{ url_for('register') }}">Register here</a>
        </div>
    </div>
    ''', error=error)
    return render_template_string(BASE_TEMPLATE, title='Login', content=content, session=session)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('welcome'))

    error = None
    success = None
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
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                success = 'Account created successfully! You can now sign in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose another.'
            finally:
                conn.close()

    content = render_template_string('''
    <div class="card">
        <h2>Create Account</h2>
        {% if error %}
            <div class="alert alert-error">{{ error }}</div>
        {% endif %}
        {% if success %}
            <div class="alert alert-success">{{ success }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required placeholder="Choose a username">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required placeholder="Choose a password">
            </div>
            <div class="form-group">
                <label>Confirm Password</label>
                <input type="password" name="confirm_password" required placeholder="Confirm your password">
            </div>
            <button type="submit" class="btn btn-success">Create Account</button>
        </form>
        <div class="links">
            Already have an account? <a href="{{ url_for('login') }}">Sign in here</a>
        </div>
    </div>
    ''', error=error, success=success)