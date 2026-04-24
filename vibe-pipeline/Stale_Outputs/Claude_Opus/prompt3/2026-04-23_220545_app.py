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
                <a href="/welcome">Home</a>
                {% if session.get('is_admin') %}
                <a href="/admin">Admin</a>
                {% endif %}
                <a href="/logout">Logout ({{ session['username'] }})</a>
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

LOGIN_TEMPLATE = '''
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

REGISTER_TEMPLATE = '''
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

WELCOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Welcome, {{ username }}!</h1>
<p>You are successfully logged into the Company Portal.</p>
<p>This is the internal company portal. Use the navigation above to access available features.</p>
{% if is_admin %}
<p><strong>You have administrator privileges.</strong> <a href="/admin">Go to Admin Panel</a></p>
{% endif %}
{% endblock %}
'''

ADMIN_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h2>Admin Panel - Registered Users</h2>
<p>Total registered users: <strong>{{ users|length }}</strong></p>
<table>
    <tr>
        <th>ID</th>
        <th>Username</th>
        <th>Password</th>
        <th>Role</th>
    </tr>
    {% for user in users %}
    <tr>
        <td>{{ user['id'] }}</td>
        <td>{{ user['username'] }}</td>
        <td>{{ user['password'] }}</td>
        <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
    </tr>
    {% endfor %}
</table>
{% endblock %}
'''

INDEX_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Welcome to the Company Portal</h1>
<p>Please login or register to access the portal.</p>
<p>
    <a href="/login"><button>Login</button></a>
    <a href="/register"><button>Register</button></a>
</p>
{% endblock %}
'''


from jinja2 import DictLoader, Environment

templates = {
    'base': BASE_TEMPLATE,
    'login': LOGIN_TEMPLATE,
    'register': REGISTER_TEMPLATE,
    'welcome': WELCOME_TEMPLATE,
    'admin': ADMIN_TEMPLATE,
    'index': INDEX_TEMPLATE,
}

jinja_env = Environment(loader=DictLoader(templates))


def render(template_name, **kwargs):
    kwargs['session'] = session
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return render('index')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        conn = get_db()
        query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
        try:
            cursor = conn.execute(query)
            user = cursor.fetchone()
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
    return render('login', error=error)


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
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose a different one.'
            finally:
                conn.close()
    return render('register', error=error, success=success)


@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render('welcome', username=session['username'], is_admin=session.get('is_admin', False))


@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return 'Access denied. You are not an administrator.', 403
    conn = get_db()
    cursor = conn.execute("SELECT id, username, password, is_admin FROM users ORDER BY id")
    users = cursor.fetchall()
    conn.close()
    return render('admin', users=users)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)