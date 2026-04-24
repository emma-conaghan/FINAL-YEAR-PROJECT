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
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f2f5; min-height: 100vh; }
        .navbar { background: #1a73e8; padding: 15px 30px; color: white; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .navbar a { color: white; text-decoration: none; margin-left: 20px; font-weight: 500; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 60px auto; padding: 40px; background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
        h1 { color: #1a73e8; margin-bottom: 10px; }
        h2 { color: #333; margin-bottom: 20px; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 6px; font-weight: 600; color: #555; }
        .form-group input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 15px; transition: border-color 0.3s; }
        .form-group input:focus { outline: none; border-color: #1a73e8; }
        .btn { background: #1a73e8; color: white; padding: 12px 30px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; width: 100%; font-weight: 600; }
        .btn:hover { background: #1557b0; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .message { padding: 12px; border-radius: 8px; margin-bottom: 20px; }
        .error { background: #fee; color: #c00; border: 1px solid #fcc; }
        .success { background: #efe; color: #070; border: 1px solid #cfc; }
        .link { text-align: center; margin-top: 20px; }
        .link a { color: #1a73e8; text-decoration: none; font-weight: 500; }
        .link a:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; color: #555; font-weight: 600; }
        tr:hover { background: #f8f9fa; }
        .badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge-admin { background: #fff3cd; color: #856404; }
        .badge-user { background: #d4edda; color: #155724; }
        .welcome-card { text-align: center; padding: 20px; }
        .welcome-card .avatar { width: 80px; height: 80px; background: #1a73e8; color: white; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 36px; font-weight: 700; margin-bottom: 20px; }
    </style>
</head>
<body>
    {% if show_nav %}
    <div class="navbar">
        <div><strong>🏢 Company Portal</strong></div>
        <div>
            <span>Hello, {{ session.get('username', 'Guest') }}</span>
            {% if session.get('is_admin') %}
            <a href="/admin">Admin Panel</a>
            {% endif %}
            <a href="/welcome">Home</a>
            <a href="/logout">Logout</a>
        </div>
    </div>
    {% endif %}
    <div class="container">
        {{ content }}
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

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
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                success = 'Registration successful! You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose another.'
            finally:
                conn.close()

    content = '''
    <h1>📋 Register</h1>
    <p style="color: #666; margin-bottom: 25px;">Create your company portal account</p>
    '''
    if error:
        content += f'<div class="message error">{error}</div>'
    if success:
        content += f'<div class="message success">{success}</div>'
    content += '''
    <form method="POST">
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
    <div class="link">
        <p>Already have an account? <a href="/login">Log in here</a></p>
    </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Register', content=content, show_nav=False, session=session)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

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
            error = 'Invalid username or password.'

    content = '''
    <h1>🔐 Login</h1>
    <p style="color: #666; margin-bottom: 25px;">Sign in to the company portal</p>
    '''
    if error:
        content += f'<div class="message error">{error}</div>'
    content += '''
    <form method="POST">
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
    <div class="link">
        <p>Don't have an account? <a href="/register">Register here</a></p>
    </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Login', content=content, show_nav=False, session=session)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    initial = username[0].upper()
    is_admin = session.get('is_admin', False)

    admin_link = ''
    if is_admin:
        admin_link = '<a href="/admin" class="btn" style="display:inline-block; width:auto; margin-top:15px; text-decoration:none;">🛠 Go to Admin Panel</a>'

    content = f'''
    <div class="welcome-card">
        <div class="avatar">{initial}</div>
        <h1>Welcome, {username}!</h1>
        <p style="color: #666; margin-top: 10px; font-size: 17px;">You are successfully logged into the Company Portal.</p>
        <p style="color: #999; margin-top: 8px;">Role: <span class="badge {'badge-admin' if is_admin else 'badge-user'}">{'Administrator' if is_admin else 'User'}</span></p>
        {admin_link}
        <hr style="margin: 30px 0; border: none; border-top: 1px solid #eee;">
        <p style="color: #888;">This is your dashboard. More features coming soon!</p>
    </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Welcome', content=content, show_nav=True, session=session)

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin', False):
        content = '''
        <h1>⛔ Access Denied</h1>
        <p style="color: #666; margin-top: 10px;">You do not have permission to access the admin area.</p>
        <a href="/welcome" class="btn" style="display:inline-block; width:auto; margin-top:20px; text-decoration:none;">Back to Home</a>
        '''
        return render_template_string(BASE_TEMPLATE, title='Access Denied', content=content, show_nav=True, session=session), 403

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()

    rows = ''
    for user in users:
        role_badge = '<span class="badge badge-admin">Admin</span>' if user['is_admin'] else '<span class="badge badge-user">User</span>'
        rows += f'<tr><td>{user["id"]}</td><td>{user["username"]}</td><td>{role_badge}</td></tr>'

    content = f'''
    <h1>🛠 Admin Panel</h1>
    <p style="color: #666; margin-bottom: 5px;">Manage all registered users</p>
    <p style="color: #999; margin-bottom: 20px;">Total users: {len(users)}</p>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Role</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    '''
    return render_template_string(BASE_TEMPLATE, title='Admin Panel', content=content, show_nav=True, session=session)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)