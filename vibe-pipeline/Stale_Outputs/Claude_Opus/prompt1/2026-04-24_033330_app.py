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
        .navbar a { color: white; text-decoration: none; margin-left: 20px; padding: 8px 16px; border-radius: 4px; transition: background 0.3s; }
        .navbar a:hover { background: rgba(255,255,255,0.2); }
        .container { max-width: 600px; margin: 60px auto; padding: 40px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .container h2 { margin-bottom: 25px; color: #333; text-align: center; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 6px; color: #555; font-weight: 600; }
        .form-group input { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 6px; font-size: 15px; transition: border-color 0.3s; }
        .form-group input:focus { outline: none; border-color: #1a73e8; }
        .btn { width: 100%; padding: 12px; background: #1a73e8; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; transition: background 0.3s; }
        .btn:hover { background: #1557b0; }
        .message { padding: 12px; border-radius: 6px; margin-bottom: 20px; text-align: center; }
        .error { background: #fce4e4; color: #cc0033; border: 1px solid #cc0033; }
        .success { background: #e4fce4; color: #006600; border: 1px solid #006600; }
        .link { text-align: center; margin-top: 20px; }
        .link a { color: #1a73e8; text-decoration: none; }
        .link a:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; color: #555; font-weight: 600; }
        tr:hover { background: #f8f9fa; }
        .badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge-admin { background: #e8f0fe; color: #1a73e8; }
        .badge-user { background: #e6f4ea; color: #137333; }
        .welcome-card { text-align: center; }
        .welcome-card h1 { font-size: 28px; color: #333; margin-bottom: 10px; }
        .welcome-card p { color: #666; font-size: 16px; }
    </style>
</head>
<body>
    <div class="navbar">
        <strong>Company Portal</strong>
        <div>
            {% if session.get('username') %}
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
    message = ''
    msg_class = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not username or not password:
            message = 'Username and password are required.'
            msg_class = 'error'
        elif password != confirm_password:
            message = 'Passwords do not match.'
            msg_class = 'error'
        elif len(password) < 4:
            message = 'Password must be at least 4 characters.'
            msg_class = 'error'
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login', registered='1'))
            except sqlite3.IntegrityError:
                message = 'Username already exists.'
                msg_class = 'error'
                conn.close()

    content = '''
        <h2>Create Account</h2>
        {% if message %}
        <div class="message {{ msg_class }}">{{ message }}</div>
        {% endif %}
        <form method="POST">
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
            <button type="submit" class="btn">Register</button>
        </form>
        <div class="link">
            <p>Already have an account? <a href="/login">Login here</a></p>
        </div>
    '''
    return render_template_string(BASE_TEMPLATE + '{% block body %}{% endblock %}',
                                  title='Register',
                                  content=render_template_string(content, message=message, msg_class=msg_class),
                                  session=session)

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    msg_class = ''

    if request.args.get('registered'):
        message = 'Registration successful! Please log in.'
        msg_class = 'success'

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                            (username, password)).fetchone()
        conn.close()

        if user:
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            session['user_id'] = user['id']
            return redirect(url_for('welcome'))
        else:
            message = 'Invalid username or password.'
            msg_class = 'error'

    content = '''
        <h2>Login</h2>
        {% if message %}
        <div class="message {{ msg_class }}">{{ message }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required placeholder="Enter your username">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required placeholder="Enter your password">
            </div>
            <button type="submit" class="btn">Login</button>
        </form>
        <div class="link">
            <p>Don't have an account? <a href="/register">Register here</a></p>
        </div>
    '''
    return render_template_string(BASE_TEMPLATE,
                                  title='Login',
                                  content=render_template_string(content, message=message, msg_class=msg_class),
                                  session=session)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))

    content = '''
        <div class="welcome-card">
            <h1>Welcome, {{ username }}!</h1>
            <p>You are successfully logged into the Company Portal.</p>
            <br>
            <p style="color: #999; font-size: 14px;">
                {% if is_admin %}
                You have administrator privileges. <a href="/admin">Go to Admin Panel</a>
                {% else %}
                You are logged in as a regular user.
                {% endif %}
            </p>
        </div>
    '''
    return render_template_string(BASE_TEMPLATE,
                                  title='Welcome',
                                  content=render_template_string(content,
                                                                  username=session['username'],
                                                                  is_admin=session.get('is_admin')),
                                  session=session)

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        content = '''
            <h2>Access Denied</h2>
            <div class="message error">You do not have permission to access the admin area.</div>
            <div class="link"><a href="/welcome">Back to Home</a></div>
        '''
        return render_template_string(BASE_TEMPLATE,
                                      title='Access Denied',
                                      content=content,
                                      session=session), 403

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()

    rows = ''
    for user in users:
        badge = '<span class="badge badge-admin">Admin</span>' if user['is_admin'] else '<span class="badge badge-user">User</span>'
        rows += '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(user['id'], user['username'], badge)

    content = '''
        <h2>Admin Panel - Registered Users</h2>
        <p style="color: #666; margin-bottom: 10px;">Total users: {}</p>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Username</th>
                    <th>Role</th>
                </tr>
            </thead>
            <tbody>
                {}
            </tbody>
        </table>
    '''.format(len(users), rows)

    return render_template_string(BASE_TEMPLATE,
                                  title='Admin',
                                  content=content,
                                  session=session)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)