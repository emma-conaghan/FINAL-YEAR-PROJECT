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
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; flex-direction: column; }
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 50px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button, .btn { background: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 14px; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .message { padding: 10px 15px; border-radius: 4px; margin-bottom: 15px; }
        .error { background: #fce4e4; color: #c0392b; border: 1px solid #f5c6cb; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .links { margin-top: 15px; }
        .links a { color: #3498db; }
        .welcome-box { text-align: center; }
        .welcome-box h1 { font-size: 28px; }
        .welcome-box p { color: #666; font-size: 16px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="navbar">
        <strong>Company Portal</strong>
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
        {% if error %}
            <div class="message error">{{ error }}</div>
        {% endif %}
        {% if success %}
            <div class="message success">{{ success }}</div>
        {% endif %}
        {{ content }}
    </div>
</body>
</html>
'''

LOGIN_CONTENT = '''
<h2>Login</h2>
<form method="POST" action="/login">
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
'''

REGISTER_CONTENT = '''
<h2>Register</h2>
<form method="POST" action="/register">
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

WELCOME_CONTENT = '''
<div class="welcome-box">
    <h1>Welcome, {{ username }}!</h1>
    <p>You are successfully logged into the Company Portal.</p>
    <p>This is the internal company portal for authorized employees.</p>
    {% if is_admin %}
        <p><a href="/admin" class="btn">Go to Admin Panel</a></p>
    {% endif %}
</div>
'''

ADMIN_CONTENT = '''
<h2>Admin Panel - Registered Users</h2>
<p>Total users: {{ users|length }}</p>
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
'''

def render_page(title, content_template, error=None, success=None, **kwargs):
    from jinja2 import Template
    content_rendered = Template(content_template).render(**kwargs)
    return render_template_string(BASE_TEMPLATE, title=title, content=content_rendered,
                                  error=error, success=success, session=session)

@app.route('/')
def index():
    if session.get('username'):
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return render_page('Login', LOGIN_CONTENT, error='Please fill in all fields.')
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                            (username, password)).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            return render_page('Login', LOGIN_CONTENT, error='Invalid username or password.')
    return render_page('Login', LOGIN_CONTENT)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not username or not password or not confirm_password:
            return render_page('Register', REGISTER_CONTENT, error='Please fill in all fields.')
        if len(username) < 3:
            return render_page('Register', REGISTER_CONTENT, error='Username must be at least 3 characters.')
        if len(password) < 4:
            return render_page('Register', REGISTER_CONTENT, error='Password must be at least 4 characters.')
        if password != confirm_password:
            return render_page('Register', REGISTER_CONTENT, error='Passwords do not match.')
        conn = get_db()
        existing = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            conn.close()
            return render_page('Register', REGISTER_CONTENT, error='Username already exists.')
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     (username, password, 0))
        conn.commit()
        conn.close()
        return render_page('Login', LOGIN_CONTENT, success='Registration successful! Please log in.')
    return render_page('Register', REGISTER_CONTENT)

@app.route('/welcome')
def welcome():
    if not session.get('username'):
        return redirect(url_for('login'))
    return render_page('Welcome', WELCOME_CONTENT,
                       username=session['username'],
                       is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if not session.get('username'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return render_page('Access Denied', '<h2>Access Denied</h2><p>You do not have permission to view this page.</p>',
                           error='Admin access required.')
    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()
    return render_page('Admin Panel', ADMIN_CONTENT, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)