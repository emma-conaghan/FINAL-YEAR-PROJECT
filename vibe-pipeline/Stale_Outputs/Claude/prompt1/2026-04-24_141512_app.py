from flask import Flask, request, redirect, url_for, session, render_template_string
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DB_PATH = 'portal.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )''')
    conn.commit()
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ('admin', admin_password))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }}
        .container {{ max-width: 500px; margin: 80px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #444; }}
        input[type=text], input[type=password] {{ width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }}
        button, .btn {{ background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; margin-top: 10px; }}
        button:hover, .btn:hover {{ background: #45a049; }}
        .btn-red {{ background: #e74c3c; }}
        .btn-red:hover {{ background: #c0392b; }}
        .btn-blue {{ background: #3498db; }}
        .btn-blue:hover {{ background: #2980b9; }}
        .error {{ color: red; margin: 10px 0; }}
        .success {{ color: green; margin: 10px 0; }}
        .nav {{ text-align: right; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
    </style>
</head>
<body>
<div class="container">
{content}
</div>
</body>
</html>
'''

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        hashed = hash_password(password)
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed)).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    content = f'''
        <h1>Company Portal</h1>
        <h2>Login</h2>
        <div class="error">{error}</div>
        <form method="post">
            <label>Username:</label>
            <input type="text" name="username" required>
            <label>Password:</label>
            <input type="password" name="password" required>
            <button type="submit">Login</button>
        </form>
        <p>Don't have an account? <a href="{url_for('register')}">Register here</a></p>
    '''
    return render_template_string(BASE_TEMPLATE.format(content=content))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    success = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm', '').strip()
        if not username or not password:
            error = 'Username and password are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        else:
            hashed = hash_password(password)
            try:
                conn = get_db()
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                conn.commit()
                conn.close()
                success = 'Registration successful! You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
    content = f'''
        <h1>Company Portal</h1>
        <h2>Register</h2>
        <div class="error">{error}</div>
        <div class="success">{success}</div>
        <form method="post">
            <label>Username:</label>
            <input type="text" name="username" required>
            <label>Password:</label>
            <input type="password" name="password" required>
            <label>Confirm Password:</label>
            <input type="password" name="confirm" required>
            <button type="submit">Register</button>
        </form>
        <p>Already have an account? <a href="{url_for('login')}">Login here</a></p>
    '''
    return render_template_string(BASE_TEMPLATE.format(content=content))

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    is_admin = session.get('is_admin', 0)
    admin_link = f'<a href="{url_for("admin")}" class="btn btn-blue">Admin Panel</a>' if is_admin else ''
    content = f'''
        <h1>Welcome, {username}!</h1>
        <p>You are logged into the Company Internal Portal.</p>
        <p>Use the navigation below to access available features.</p>
        <div class="nav">
            {admin_link}
            <a href="{url_for('logout')}" class="btn btn-red">Logout</a>
        </div>
        <hr>
        <h2>Company Announcements</h2>
        <p>Welcome to the internal portal. Stay tuned for updates and announcements from management.</p>
    '''
    return render_template_string(BASE_TEMPLATE.format(content=content))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return render_template_string(BASE_TEMPLATE.format(content='<h2>Access Denied</h2><p>You do not have permission to view this page.</p><a href="/welcome" class="btn">Go Back</a>'))
    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
    conn.close()
    rows = ''
    for u in users:
        role = 'Admin' if u['is_admin'] else 'User'
        rows += f'<tr><td>{u["id"]}</td><td>{u["username"]}</td><td>{role}</td></tr>'
    content = f'''
        <h1>Admin Panel</h1>
        <div class="nav">
            <a href="{url_for('welcome')}" class="btn">Back to Welcome</a>
            <a href="{url_for('logout')}" class="btn btn-red">Logout</a>
        </div>
        <h2>Registered Users</h2>
        <table>
            <tr><th>ID</th><th>Username</th><th>Role</th></tr>
            {rows}
        </table>
    '''
    return render_template_string(BASE_TEMPLATE.format(content=content))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)