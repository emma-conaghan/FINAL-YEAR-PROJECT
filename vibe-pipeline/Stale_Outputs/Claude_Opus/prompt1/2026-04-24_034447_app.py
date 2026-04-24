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
        .navbar a { color: white; text-decoration: none; margin-left: 20px; padding: 8px 16px; border-radius: 4px; transition: background 0.2s; }
        .navbar a:hover { background: rgba(255,255,255,0.2); }
        .container { max-width: 600px; margin: 60px auto; padding: 0 20px; }
        .card { background: white; border-radius: 12px; padding: 40px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
        .card h2 { color: #1a73e8; margin-bottom: 25px; text-align: center; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 6px; color: #555; font-weight: 500; }
        .form-group input { width: 100%; padding: 12px 16px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 15px; transition: border-color 0.2s; }
        .form-group input:focus { outline: none; border-color: #1a73e8; }
        .btn { display: inline-block; padding: 12px 32px; background: #1a73e8; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; width: 100%; transition: background 0.2s; text-align: center; text-decoration: none; }
        .btn:hover { background: #1557b0; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #b02a37; }
        .message { padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; }
        .message.error { background: #fce4e4; color: #cc0033; border: 1px solid #f5c6c6; }
        .message.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .link-text { text-align: center; margin-top: 20px; color: #666; }
        .link-text a { color: #1a73e8; text-decoration: none; font-weight: 500; }
        .link-text a:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { background: #f8f9fa; color: #555; font-weight: 600; }
        tr:hover { background: #f8f9fa; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge-admin { background: #fff3cd; color: #856404; }
        .badge-user { background: #d4edda; color: #155724; }
        .welcome-section { text-align: center; }
        .welcome-section h1 { color: #333; font-size: 28px; margin-bottom: 10px; }
        .welcome-section p { color: #666; font-size: 16px; line-height: 1.6; }
        .wide-container { max-width: 900px; margin: 60px auto; padding: 0 20px; }
    </style>
</head>
<body>
    <div class="navbar">
        <strong>🏢 Company Portal</strong>
        <div>
            {% if session.get('username') %}
                <span>Hello, {{ session['username'] }}</span>
                <a href="/welcome">Home</a>
                {% if session.get('is_admin') %}
                    <a href="/admin">Admin</a>
                {% endif %}
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
{% block content %}
<div class="container">
    <div class="card">
        <h2>🔐 Login</h2>
        {% if error %}
            <div class="message error">{{ error }}</div>
        {% endif %}
        {% if success %}
            <div class="message success">{{ success }}</div>
        {% endif %}
        <form method="POST" action="/login">
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
        <div class="link-text">
            <p>Don't have an account? <a href="/register">Register here</a></p>
        </div>
    </div>
</div>
{% endblock %}
'''

REGISTER_PAGE = '''
{% extends "base" %}
{% block content %}
<div class="container">
    <div class="card">
        <h2>📝 Register</h2>
        {% if error %}
            <div class="message error">{{ error }}</div>
        {% endif %}
        <form method="POST" action="/register">
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
            <button type="submit" class="btn">Create Account</button>
        </form>
        <div class="link-text">
            <p>Already have an account? <a href="/login">Login here</a></p>
        </div>
    </div>
</div>
{% endblock %}
'''

WELCOME_PAGE = '''
{% extends "base" %}
{% block content %}
<div class="container">
    <div class="card welcome-section">
        <h1>👋 Welcome, {{ session['username'] }}!</h1>
        <p>You are successfully logged into the Company Portal.</p>
        <br>
        <p>This is your internal company dashboard. Use the navigation bar above to access different sections.</p>
        {% if session.get('is_admin') %}
            <br>
            <p><strong>🛡️ You have administrator privileges.</strong></p>
            <br>
            <a href="/admin" class="btn" style="width: auto; display: inline-block;">Go to Admin Panel</a>
        {% endif %}
    </div>
</div>
{% endblock %}
'''

ADMIN_PAGE = '''
{% extends "base" %}
{% block content %}
<div class="wide-container">
    <div class="card">
        <h2>🛡️ Admin Panel - Registered Users</h2>
        <p style="text-align:center; color:#666; margin-bottom:20px;">Total users: {{ users|length }}</p>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Username</th>
                    <th>Password</th>
                    <th>Role</th>
                    <th>Created At</th>
                </tr>
            </thead>
            <tbody>
                {% for user in users %}
                <tr>
                    <td>{{ user['id'] }}</td>
                    <td>{{ user['username'] }}</td>
                    <td>{{ user['password'] }}</td>
                    <td>
                        {% if user['is_admin'] %}
                            <span class="badge badge-admin">Admin</span>
                        {% else %}
                            <span class="badge badge-user">User</span>
                        {% endif %}
                    </td>
                    <td>{{ user['created_at'] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>
{% endblock %}
'''

INDEX_PAGE = '''
{% extends "base" %}
{% block content %}
<div class="container">
    <div class="card welcome-section">
        <h1>🏢 Welcome to the Company Portal</h1>
        <br>
        <p>This is the internal company portal. Please log in or register to continue.</p>
        <br>
        <div style="display: flex; gap: 15px; justify-content: center;">
            <a href="/login" class="btn" style="width: auto;">Login</a>
            <a href="/register" class="btn" style="width: auto; background: #28a745;">Register</a>
        </div>
    </div>
</div>
{% endblock %}
'''

def render_page(template_content, title="Portal", **kwargs):
    full_template = BASE_TEMPLATE.replace('{{ content }}', '{% block content %}{% endblock %}')
    full_template = full_template.replace('{% extends "base" %}', '')
    combined = BASE_TEMPLATE.replace('{{ content }}', template_content.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''))
    return render_template_string(combined, title=title, session=session, **kwargs)


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return render_page(INDEX_PAGE.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''), title="Home")


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    success = request.args.get('success')
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
            session['is_admin'] = user['is_admin']
            session['user_id'] = user['id']
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    content = LOGIN_PAGE.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', '')
    return render_page(content, title="Login", error=error, success=success)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
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
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login', success='Registration successful! Please log in.'))
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose another.'
                conn.close()
    content = REGISTER_PAGE.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', '')
    return render_page(content, title="Register", error=error)


@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    content = WELCOME_PAGE.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', '')
    return render_page(content, title="Welcome")


@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))
    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()
    content = ADMIN_PAGE.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', '')
    return render_page(content, title="Admin", users=users)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)