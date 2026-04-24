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
    <title>{{ title }} - Company Portal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; flex-direction: column; }
        nav { background: #2c3e50; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        nav a { color: white; text-decoration: none; margin-left: 15px; font-size: 14px; }
        nav a:hover { text-decoration: underline; }
        nav .brand { color: white; font-size: 20px; font-weight: bold; }
        .container { max-width: 500px; margin: 60px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .container.wide { max-width: 800px; }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; }
        label { display: block; margin-bottom: 5px; color: #555; font-weight: bold; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button, .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .error { background: #fee; color: #c0392b; padding: 10px; border-radius: 4px; margin-bottom: 15px; border: 1px solid #fcc; }
        .success { background: #efe; color: #27ae60; padding: 10px; border-radius: 4px; margin-bottom: 15px; border: 1px solid #cfc; }
        .info { background: #eef; color: #2c3e50; padding: 10px; border-radius: 4px; margin-bottom: 15px; border: 1px solid #ccf; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .welcome-box { text-align: center; padding: 20px; }
        .welcome-box h1 { font-size: 28px; margin-bottom: 10px; }
        .welcome-box p { color: #666; font-size: 16px; margin-bottom: 20px; }
        .links { margin-top: 10px; }
        .links a { margin: 0 5px; }
    </style>
</head>
<body>
    <nav>
        <span class="brand">Company Portal</span>
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
    </nav>
    <div class="container {{ container_class|default('') }}">
        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}
        {% if success %}
            <div class="success">{{ success }}</div>
        {% endif %}
        {{ content }}
    </div>
</body>
</html>
'''

LOGIN_CONTENT = '''
<h2>Login</h2>
<form method="POST">
    <label for="username">Username</label>
    <input type="text" id="username" name="username" required placeholder="Enter your username">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" required placeholder="Enter your password">
    <button type="submit">Login</button>
</form>
<div class="links">
    <p style="margin-top: 15px; color: #666;">Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
</div>
'''

REGISTER_CONTENT = '''
<h2>Create Account</h2>
<form method="POST">
    <label for="username">Username</label>
    <input type="text" id="username" name="username" required placeholder="Choose a username">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" required placeholder="Choose a password">
    <label for="confirm_password">Confirm Password</label>
    <input type="password" id="confirm_password" name="confirm_password" required placeholder="Confirm your password">
    <button type="submit">Register</button>
</form>
<div class="links">
    <p style="margin-top: 15px; color: #666;">Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
</div>
'''

WELCOME_CONTENT = '''
<div class="welcome-box">
    <h1>Welcome, {{ session.get('username') }}!</h1>
    <p>You are logged into the Company Portal.</p>
    <div class="info">
        <strong>Your Account Details:</strong><br>
        Username: {{ session.get('username') }}<br>
        Role: {{ "Administrator" if session.get('is_admin') else "User" }}
    </div>
    {% if session.get('is_admin') %}
        <a href="{{ url_for('admin_panel') }}" class="btn">Go to Admin Panel</a>
    {% endif %}
</div>
'''

ADMIN_CONTENT = '''
<h2>Admin Panel - Registered Users</h2>
<p style="color: #666; margin-bottom: 15px;">Total users: {{ users|length }}</p>
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Role</th>
        </tr>
    </thead>
    <tbody>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ "Admin" if user['is_admin'] else "User" }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
'''

@app.route('/')
def index():
    if session.get('user_id'):
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Please fill in all fields.'
        else:
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                                (username, password)).fetchone()
            conn.close()
            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = bool(user['is_admin'])
                return redirect(url_for('welcome'))
            else:
                error = 'Invalid username or password.'
    return render_template_string(BASE_TEMPLATE,
                                  title='Login',
                                  content=render_template_string(LOGIN_CONTENT),
                                  error=error,
                                  success=request.args.get('success'),
                                  session=session)

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
            existing = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                error = 'Username already exists. Please choose another.'
                conn.close()
            else:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login', success='Registration successful! Please log in.'))
    return render_template_string(BASE_TEMPLATE,
                                  title='Register',
                                  content=render_template_string(REGISTER_CONTENT),
                                  error=error,
                                  success=None,
                                  session=session)

@app.route('/welcome')
def welcome():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template_string(BASE_TEMPLATE,
                                  title='Welcome',
                                  content=render_template_string(WELCOME_CONTENT, session=session),
                                  error=None,
                                  success=None,
                                  session=session)

@app.route('/admin')
def admin_panel():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return render_template_string(BASE_TEMPLATE,
                                      title='Access Denied',
                                      content='<h2>Access Denied</h2><p>You do not have permission to view this page.</p><a href="{{ url_for("welcome") }}" class="btn">Go Back</a>',
                                      error='You must be an administrator to access this page.',
                                      success=None,
                                      session=session)
    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()
    return render_template_string(BASE_TEMPLATE,
                                  title='Admin Panel',
                                  content=render_template_string(ADMIN_CONTENT, users=users),
                                  error=None,
                                  success=None,
                                  session=session,
                                  container_class='wide')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login', success='You have been logged out successfully.'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)