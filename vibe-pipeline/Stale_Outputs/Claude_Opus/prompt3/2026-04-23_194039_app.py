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
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES ('admin', 'admin', 1)")
    conn.commit()
    conn.close()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f6f9; }
        .navbar { background: #2c3e50; padding: 15px 30px; color: white; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 60px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin: 8px 0 16px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button, input[type="submit"] { background: #2c3e50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover, input[type="submit"]:hover { background: #34495e; }
        .error { color: red; margin-bottom: 10px; }
        .success { color: green; margin-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .links { margin-top: 15px; }
        .links a { color: #2c3e50; }
    </style>
</head>
<body>
    <div class="navbar">
        <span><strong>Company Portal</strong></span>
        <div>
            {% if session.get('username') %}
                <span>Hello, {{ session['username'] }}</span>
                {% if session.get('is_admin') %}
                    <a href="/admin">Admin</a>
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
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

HOME_PAGE = '''
{% extends "base" %}
{% block content %}
<h1>Welcome to the Company Portal</h1>
<p>Please login or register to access the portal.</p>
<a href="/login"><button>Login</button></a>
<a href="/register"><button>Register</button></a>
{% endblock %}
'''

LOGIN_PAGE = '''
{% extends "base" %}
{% block content %}
<h2>Login</h2>
{% if error %}
<p class="error">{{ error }}</p>
{% endif %}
<form method="POST" action="/login">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Password:</label>
    <input type="password" name="password" required>
    <input type="submit" value="Login">
</form>
<div class="links">
    <p>Don't have an account? <a href="/register">Register here</a></p>
</div>
{% endblock %}
'''

REGISTER_PAGE = '''
{% extends "base" %}
{% block content %}
<h2>Register</h2>
{% if error %}
<p class="error">{{ error }}</p>
{% endif %}
{% if success %}
<p class="success">{{ success }}</p>
{% endif %}
<form method="POST" action="/register">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Password:</label>
    <input type="password" name="password" required>
    <label>Confirm Password:</label>
    <input type="password" name="confirm_password" required>
    <input type="submit" value="Register">
</form>
<div class="links">
    <p>Already have an account? <a href="/login">Login here</a></p>
</div>
{% endblock %}
'''

WELCOME_PAGE = '''
{% extends "base" %}
{% block content %}
<h1>Welcome, {{ username }}!</h1>
<p>You are now logged into the Company Portal.</p>
<p>This is your internal dashboard. Use the navigation above to access different areas.</p>
{% if is_admin %}
<p><strong>You have administrator privileges.</strong></p>
<a href="/admin"><button>Go to Admin Panel</button></a>
{% endif %}
{% endblock %}
'''

ADMIN_PAGE = '''
{% extends "base" %}
{% block content %}
<h2>Admin Panel - Registered Users</h2>
<p>Total users: {{ users|length }}</p>
<table>
    <tr>
        <th>ID</th>
        <th>Username</th>
        <th>Role</th>
    </tr>
    {% for user in users %}
    <tr>
        <td>{{ user['id'] }}</td>
        <td>{{ user['username'] }}</td>
        <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
    </tr>
    {% endfor %}
</table>
{% endblock %}
'''

def render_with_base(template_string, **kwargs):
    from jinja2 import BaseLoader, Environment
    env = Environment(loader=BaseLoader())
    base_template = env.from_string(BASE_TEMPLATE)
    env.globals['session'] = session
    combined = BASE_TEMPLATE.replace('{% block content %}{% endblock %}',
                                      template_string.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''))
    template = env.from_string(combined)
    return template.render(session=session, **kwargs)

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return render_with_base(HOME_PAGE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_with_base(LOGIN_PAGE, error=error)

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
        elif len(password) < 3:
            error = 'Password must be at least 3 characters.'
        else:
            conn = get_db()
            existing = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                error = 'Username already exists.'
                conn.close()
            else:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                conn.close()
                success = 'Registration successful! You can now login.'
    return render_with_base(REGISTER_PAGE, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_with_base(WELCOME_PAGE, username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin', False):
        return render_with_base('<h2>Access Denied</h2><p>You do not have permission to access this page.</p>')
    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()
    return render_with_base(ADMIN_PAGE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)