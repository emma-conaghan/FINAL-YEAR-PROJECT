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
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
        .navbar { width: 100%; background: #2c3e50; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .navbar .brand { font-size: 1.3em; font-weight: bold; color: #ecf0f1; }
        .container { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 40px; width: 90%; max-width: 500px; }
        .container.wide { max-width: 800px; }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 1em; }
        button, .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .error { background: #fdecea; color: #c0392b; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .success { background: #eafaf1; color: #27ae60; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .info { color: #777; margin-top: 15px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; color: #2c3e50; }
        tr:hover { background: #f8f9fa; }
        .welcome-box { text-align: center; }
        .welcome-box h1 { font-size: 2em; }
        .welcome-box p { font-size: 1.2em; color: #666; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="navbar">
        <span class="brand">🏢 Company Portal</span>
        <div>
            {% if session.get('username') %}
                <a href="{{ url_for('welcome') }}">Home</a>
                {% if session.get('is_admin') %}
                    <a href="{{ url_for('admin_panel') }}">Admin</a>
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout ({{ session['username'] }})</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </div>
    {% block content %}{% endblock %}
</body>
</html>
'''

INDEX_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="container welcome-box">
    <h1>Welcome to the Company Portal</h1>
    <p>Please login or register to access the portal.</p>
    <a href="{{ url_for('login') }}" class="btn">Login</a>
    <a href="{{ url_for('register') }}" class="btn" style="background:#27ae60;">Register</a>
</div>
{% endblock %}
'''

LOGIN_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="container">
    <h2>Login</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    {% if success %}
        <div class="success">{{ success }}</div>
    {% endif %}
    <form method="POST">
        <label for="username">Username</label>
        <input type="text" id="username" name="username" required autofocus>
        <label for="password">Password</label>
        <input type="password" id="password" name="password" required>
        <button type="submit">Login</button>
    </form>
    <p class="info">Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
</div>
{% endblock %}
'''

REGISTER_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="container">
    <h2>Register</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    <form method="POST">
        <label for="username">Username</label>
        <input type="text" id="username" name="username" required autofocus>
        <label for="password">Password</label>
        <input type="password" id="password" name="password" required>
        <label for="confirm_password">Confirm Password</label>
        <input type="password" id="confirm_password" name="confirm_password" required>
        <button type="submit" style="background:#27ae60;">Register</button>
    </form>
    <p class="info">Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
</div>
{% endblock %}
'''

WELCOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="container welcome-box">
    <h1>Welcome, {{ session['username'] }}! 👋</h1>
    <p>You are successfully logged into the Company Portal.</p>
    {% if session.get('is_admin') %}
        <p style="color:#e67e22; font-weight:bold;">You have administrator privileges.</p>
        <a href="{{ url_for('admin_panel') }}" class="btn" style="background:#e67e22;">Go to Admin Panel</a>
    {% endif %}
    <br><br>
    <a href="{{ url_for('logout') }}" class="btn btn-danger">Logout</a>
</div>
{% endblock %}
'''

ADMIN_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="container wide">
    <h2>Admin Panel - Registered Users</h2>
    <p style="color:#666; margin-bottom:15px;">Total users: {{ users|length }}</p>
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
                <td>{% if user['is_admin'] %}<strong style="color:#e67e22;">Admin</strong>{% else %}User{% endif %}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endblock %}
'''

templates = {
    'base': BASE_TEMPLATE,
    'index': INDEX_TEMPLATE,
    'login': LOGIN_TEMPLATE,
    'register': REGISTER_TEMPLATE,
    'welcome': WELCOME_TEMPLATE,
    'admin': ADMIN_TEMPLATE,
}

from jinja2 import BaseLoader, TemplateNotFound

class DictLoader(BaseLoader):
    def __init__(self, mapping):
        self.mapping = mapping

    def get_source(self, environment, template):
        if template in self.mapping:
            source = self.mapping[template]
            return source, None, lambda: True
        raise TemplateNotFound(template)

app.jinja_loader = DictLoader(templates)

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return render_template_string(INDEX_TEMPLATE, title='Home', session=session)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    success = request.args.get('success')
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Please fill in all fields.'
        else:
            conn = get_db()
            query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
            try:
                cursor = conn.execute(query)
                user = cursor.fetchone()
            except Exception as e:
                error = 'An error occurred: ' + str(e)
                user = None
            conn.close()
            if user:
                session['username'] = user['username']
                session['is_admin'] = bool(user['is_admin'])
                session['user_id'] = user['id']
                return redirect(url_for('welcome'))
            else:
                if not error:
                    error = 'Invalid username or password.'
    return render_template_string(LOGIN_TEMPLATE, title='Login', error=error, success=success, session=session)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not username or not password or not confirm_password:
            error = 'Please fill in all fields.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters long.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters long.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                             (username, password, 0))
                conn.commit()
                conn.close()
                return redirect(url_for('login', success='Registration successful! Please log in.'))
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose a different one.'
                conn.close()
    return render_template_string(REGISTER_TEMPLATE, title='Register', error=error, session=session)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, title='Welcome', session=session)

@app.route('/admin')
def admin_panel():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return 'Access denied. You are not an administrator.', 403
    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()
    return render_template_string(ADMIN_TEMPLATE, title='Admin Panel', users=users, session=session)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)