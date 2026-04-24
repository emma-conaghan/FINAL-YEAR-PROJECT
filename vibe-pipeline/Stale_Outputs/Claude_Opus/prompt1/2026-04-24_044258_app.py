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
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES ('admin', 'admin123', 1)")
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
        .navbar a { color: white; text-decoration: none; margin-left: 20px; padding: 8px 16px; border-radius: 4px; transition: background 0.3s; }
        .navbar a:hover { background: rgba(255,255,255,0.2); }
        .container { max-width: 600px; margin: 60px auto; padding: 0 20px; }
        .card { background: white; border-radius: 12px; padding: 40px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
        .card h2 { margin-bottom: 25px; color: #333; text-align: center; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 6px; color: #555; font-weight: 600; font-size: 14px; }
        .form-group input { width: 100%; padding: 12px 16px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 15px; transition: border-color 0.3s; }
        .form-group input:focus { outline: none; border-color: #1a73e8; }
        .btn { width: 100%; padding: 13px; background: #1a73e8; color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; font-weight: 600; transition: background 0.3s; }
        .btn:hover { background: #1557b0; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .message { padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; text-align: center; font-size: 14px; }
        .error { background: #fce4e4; color: #cc0033; border: 1px solid #fcc2c3; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .link-text { text-align: center; margin-top: 20px; color: #666; }
        .link-text a { color: #1a73e8; text-decoration: none; font-weight: 600; }
        .link-text a:hover { text-decoration: underline; }
        .welcome-section { text-align: center; }
        .welcome-section h1 { font-size: 32px; color: #333; margin-bottom: 10px; }
        .welcome-section p { color: #666; font-size: 18px; margin-bottom: 30px; }
        .user-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .user-table th, .user-table td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        .user-table th { background: #f8f9fa; color: #555; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
        .user-table tr:hover { background: #f8f9fa; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge-admin { background: #fff3cd; color: #856404; }
        .badge-user { background: #d4edda; color: #155724; }
        .admin-container { max-width: 900px; margin: 60px auto; padding: 0 20px; }
        .stats { display: flex; gap: 20px; margin-bottom: 30px; }
        .stat-card { flex: 1; background: white; border-radius: 12px; padding: 25px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center; }
        .stat-card h3 { font-size: 36px; color: #1a73e8; }
        .stat-card p { color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    {% if session.get('username') %}
    <div class="navbar">
        <strong>🏢 Company Portal</strong>
        <div>
            <span>Hello, {{ session['username'] }}</span>
            {% if session.get('is_admin') %}
            <a href="/admin">Admin Panel</a>
            {% endif %}
            <a href="/welcome">Home</a>
            <a href="/logout">Logout</a>
        </div>
    </div>
    {% else %}
    <div class="navbar">
        <strong>🏢 Company Portal</strong>
        <div>
            <a href="/login">Login</a>
            <a href="/register">Register</a>
        </div>
    </div>
    {% endif %}
    {{ content }}
</body>
</html>
'''

LOGIN_PAGE = '''
<div class="container">
    <div class="card">
        <h2>🔐 Sign In</h2>
        {% if error %}
        <div class="message error">{{ error }}</div>
        {% endif %}
        {% if success %}
        <div class="message success">{{ success }}</div>
        {% endif %}
        <form method="POST" action="/login">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" placeholder="Enter your username" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" placeholder="Enter your password" required>
            </div>
            <button type="submit" class="btn">Sign In</button>
        </form>
        <p class="link-text">Don't have an account? <a href="/register">Register here</a></p>
    </div>
</div>
'''

REGISTER_PAGE = '''
<div class="container">
    <div class="card">
        <h2>📝 Create Account</h2>
        {% if error %}
        <div class="message error">{{ error }}</div>
        {% endif %}
        <form method="POST" action="/register">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" placeholder="Choose a username" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" placeholder="Choose a password" required>
            </div>
            <div class="form-group">
                <label>Confirm Password</label>
                <input type="password" name="confirm_password" placeholder="Confirm your password" required>
            </div>
            <button type="submit" class="btn">Create Account</button>
        </form>
        <p class="link-text">Already have an account? <a href="/login">Sign in here</a></p>
    </div>
</div>
'''

WELCOME_PAGE = '''
<div class="container">
    <div class="card welcome-section">
        <h1>Welcome, {{ session['username'] }}! 👋</h1>
        <p>You are now logged into the Company Portal.</p>
        <p style="color: #999; font-size: 14px;">This is your internal company dashboard. More features coming soon!</p>
    </div>
</div>
'''

ADMIN_PAGE = '''
<div class="admin-container">
    <div class="stats">
        <div class="stat-card">
            <h3>{{ total_users }}</h3>
            <p>Total Users</p>
        </div>
        <div class="stat-card">
            <h3>{{ admin_count }}</h3>
            <p>Administrators</p>
        </div>
        <div class="stat-card">
            <h3>{{ total_users - admin_count }}</h3>
            <p>Regular Users</p>
        </div>
    </div>
    <div class="card">
        <h2>👥 All Registered Users</h2>
        <table class="user-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Username</th>
                    <th>Role</th>
                    <th>Registered</th>
                </tr>
            </thead>
            <tbody>
                {% for user in users %}
                <tr>
                    <td>{{ user['id'] }}</td>
                    <td><strong>{{ user['username'] }}</strong></td>
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
'''

def render_page(title, content_template, **kwargs):
    from jinja2 import Template
    content_t = Template(content_template)
    content_html = content_t.render(session=session, **kwargs)
    base_t = Template(BASE_TEMPLATE)
    return base_t.render(title=title, session=session, content=content_html)


@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('welcome'))

    error = None
    success = request.args.get('success')

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            error = 'Please fill in all fields.'
        else:
            conn = get_db()
            query = "SELECT * FROM users WHERE username = ? AND password = ?"
            user = conn.execute(query, (username, password)).fetchone()
            conn.close()

            if user:
                session['username'] = user['username']
                session['user_id'] = user['id']
                session['is_admin'] = bool(user['is_admin'])
                return redirect(url_for('welcome'))
            else:
                error = 'Invalid username or password.'

    return render_page('Login', LOGIN_PAGE, error=error, success=success)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('welcome'))

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
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login', success='Account created successfully! Please sign in.'))
            except sqlite3.IntegrityError:
                conn.close()
                error = 'Username already exists. Please choose a different one.'

    return render_page('Register', REGISTER_PAGE, error=error)


@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_page('Welcome', WELCOME_PAGE)


@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))

    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY id ASC").fetchall()
    total_users = len(users)
    admin_count = sum(1 for u in users if u['is_admin'])
    conn.close()

    return render_page('Admin Panel', ADMIN_PAGE, users=users, total_users=total_users, admin_count=admin_count)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)