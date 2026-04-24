import sqlite3
import os
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DATABASE = 'portal.db'

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
        is_admin INTEGER DEFAULT 0
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
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; flex-direction: column; }
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 50px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button, .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .error { color: #e74c3c; margin-bottom: 15px; padding: 10px; background: #fde8e8; border-radius: 4px; }
        .success { color: #27ae60; margin-bottom: 15px; padding: 10px; background: #e8fde8; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .links { margin-top: 15px; }
        .links a { color: #3498db; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
        .welcome-box { text-align: center; }
        .welcome-box h1 { font-size: 2em; }
        .welcome-box p { color: #666; font-size: 1.1em; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="navbar">
        <span><strong>Company Portal</strong></span>
        <div>
            {% if session.get('username') %}
                <span>Hello, {{ session['username'] }}</span>
                {% if session.get('is_admin') %}
                    <a href="/admin">Admin Panel</a>
                {% endif %}
                <a href="/welcome">Home</a>
                <a href="/logout">Logout</a>
            {% else %}
                <a href="/login">Login</a>
                <a href="/register">Register</a>
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
<h2>Login</h2>
{% if error %}
<div class="error">{{ error }}</div>
{% endif %}
<form method="POST">
    <div class="form-group">
        <label>Username</label>
        <input type="text" name="username" required>
    </div>
    <div class="form-group">
        <label>Password</label>
        <input type="password" name="password" required>
    </div>
    <button type="submit">Login</button>
</form>
<div class="links">
    <p>Don't have an account? <a href="/register">Register here</a></p>
</div>
{% endset %}
'''

REGISTER_PAGE = '''
<h2>Register</h2>
{% if error %}
<div class="error">{{ error }}</div>
{% endif %}
{% if success %}
<div class="success">{{ success }}</div>
{% endif %}
<form method="POST">
    <div class="form-group">
        <label>Username</label>
        <input type="text" name="username" required>
    </div>
    <div class="form-group">
        <label>Password</label>
        <input type="password" name="password" required>
    </div>
    <div class="form-group">
        <label>Confirm Password</label>
        <input type="password" name="confirm_password" required>
    </div>
    <button type="submit" class="btn-success">Register</button>
</form>
<div class="links">
    <p>Already have an account? <a href="/login">Login here</a></p>
</div>
'''

WELCOME_PAGE = '''
<div class="welcome-box">
    <h1>Welcome, {{ username }}!</h1>
    <p>You have successfully logged into the Company Portal.</p>
    <p>This is your personal dashboard.</p>
    {% if is_admin %}
    <p><a href="/admin" class="btn">Go to Admin Panel</a></p>
    {% endif %}
</div>
'''

ADMIN_PAGE = '''
<h2>Admin Panel - Registered Users</h2>
<p>Total users: {{ users|length }}</p>
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Password</th>
            <th>Role</th>
        </tr>
    </thead>
    <tbody>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ user['password'] }}</td>
            <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
<div class="links" style="margin-top: 20px;">
    <a href="/welcome" class="btn">Back to Home</a>
</div>
'''

INDEX_PAGE = '''
<div class="welcome-box">
    <h1>Welcome to the Company Portal</h1>
    <p>Please login or register to access the portal.</p>
    <p style="margin-top: 20px;">
        <a href="/login" class="btn">Login</a>
        <a href="/register" class="btn btn-success">Register</a>
    </p>
</div>
'''

def render_page(title, content_template, **kwargs):
    from jinja2 import Template
    content_tmpl = Template(content_template)
    content_html = content_tmpl.render(**kwargs)
    page_tmpl = Template(BASE_TEMPLATE)
    return page_tmpl.render(title=title, content=content_html, session=session)


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return render_page('Home', INDEX_PAGE)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        conn = get_db()
        query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
        try:
            user = conn.execute(query).fetchone()
        except Exception:
            user = None
        conn.close()
        if user:
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            session['user_id'] = user['id']
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_page('Login', LOGIN_PAGE if False else '''
<h2>Login</h2>
{% if error %}
<div class="error">{{ error }}</div>
{% endif %}
<form method="POST">
    <div class="form-group">
        <label>Username</label>
        <input type="text" name="username" required>
    </div>
    <div class="form-group">
        <label>Password</label>
        <input type="password" name="password" required>
    </div>
    <button type="submit">Login</button>
</form>
<div class="links">
    <p>Don't have an account? <a href="/register">Register here</a></p>
</div>
''', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not password:
            error = 'Username and password are required.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                             (username, password, 0))
                conn.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose a different one.'
            finally:
                conn.close()

    return render_page('Register', REGISTER_PAGE, error=error, success=success)


@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_page('Welcome', WELCOME_PAGE,
                       username=session['username'],
                       is_admin=session.get('is_admin', False))


@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return render_page('Access Denied', '<div class="error"><h2>Access Denied</h2><p>You do not have permission to access this page.</p></div>')

    conn = get_db()
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return render_page('Admin Panel', ADMIN_PAGE, users=users)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)