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
                     ('admin', 'admin', 1))
    conn.commit()
    conn.close()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f0f2f5; }
        .navbar { background: #2c3e50; padding: 15px 30px; color: white; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 50px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin: 8px 0 16px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button, input[type="submit"] { background: #2c3e50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover, input[type="submit"]:hover { background: #34495e; }
        .error { color: red; margin-bottom: 15px; }
        .success { color: green; margin-bottom: 15px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #2c3e50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .links { margin-top: 15px; }
        .links a { color: #2c3e50; }
    </style>
</head>
<body>
    <div class="navbar">
        <span><strong>Company Portal</strong></span>
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
    <p>This is the internal company portal. Please log in or register to continue.</p>
    <p><a href="/login"><button>Login</button></a> <a href="/register"><button>Register</button></a></p>
{% endblock %}
'''

LOGIN_PAGE = '''
{% extends "base" %}
{% block content %}
    <h2>Login</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
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
        <div class="error">{{ error }}</div>
    {% endif %}
    {% if success %}
        <div class="success">{{ success }}</div>
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

DASHBOARD_PAGE = '''
{% extends "base" %}
{% block content %}
    <h2>Dashboard</h2>
    <p>Hello, <strong>{{ session['username'] }}</strong>! You are now logged in.</p>
    <p>Welcome to the company portal. This is your personal dashboard.</p>
    <hr>
    <h3>Quick Links</h3>
    <ul>
        <li><a href="/profile">My Profile</a></li>
        {% if session.get('is_admin') %}
        <li><a href="/admin">Admin Panel - View All Users</a></li>
        {% endif %}
    </ul>
{% endblock %}
'''

PROFILE_PAGE = '''
{% extends "base" %}
{% block content %}
    <h2>My Profile</h2>
    <p><strong>Username:</strong> {{ user['username'] }}</p>
    <p><strong>Role:</strong> {{ "Administrator" if user['is_admin'] else "Regular User" }}</p>
    <p><strong>User ID:</strong> {{ user['id'] }}</p>
{% endblock %}
'''

ADMIN_PAGE = '''
{% extends "base" %}
{% block content %}
    <h2>Admin Panel - All Registered Users</h2>
    <p>Total users: {{ users|length }}</p>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Role</th>
            <th>Actions</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ "Admin" if user['is_admin'] else "User" }}</td>
            <td>
                {% if not user['is_admin'] %}
                <form method="POST" action="/admin/delete/{{ user['id'] }}" style="display:inline;">
                    <button type="submit" onclick="return confirm('Are you sure you want to delete this user?');" style="background:red;padding:5px 10px;font-size:12px;">Delete</button>
                </form>
                {% else %}
                    -
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
{% endblock %}
'''

from jinja2 import DictLoader, Environment

template_loader = DictLoader({
    'base': BASE_TEMPLATE,
    'home': HOME_PAGE,
    'login': LOGIN_PAGE,
    'register': REGISTER_PAGE,
    'dashboard': DASHBOARD_PAGE,
    'profile': PROFILE_PAGE,
    'admin': ADMIN_PAGE,
})

jinja_env = Environment(loader=template_loader)


def render(template_name, **kwargs):
    kwargs['session'] = session
    template = jinja_env.get_template(template_name)
    return template.render(**kwargs)


@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render('home')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
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
        elif len(username) < 3:
            error = 'Username must be at least 3 characters long.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters long.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        else:
            conn = get_db()
            existing = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                error = 'Username already exists. Please choose a different one.'
                conn.close()
            else:
                conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                           (username, password, 0))
                conn.commit()
                conn.close()
                success = 'Registration successful! You can now log in.'
    
    return render('register', error=error, success=success)


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render('dashboard')


@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    conn.close()
    
    if not user:
        return redirect(url_for('logout'))
    
    return render('profile', user=user)


@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return 'Access denied. Admins only.', 403
    
    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()
    
    return render('admin', users=users)


@app.route('/admin/delete/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return 'Access denied. Admins only.', 403
    
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user and not user['is_admin']:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    conn.close()
    
    return redirect(url_for('admin'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)