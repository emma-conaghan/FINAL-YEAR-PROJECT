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
        nav a { color: white; text-decoration: none; margin-left: 15px; }
        nav a:hover { text-decoration: underline; }
        nav .brand { font-size: 1.3em; font-weight: bold; color: white; }
        .container { max-width: 600px; margin: 50px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; }
        form { display: flex; flex-direction: column; }
        label { margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="text"], input[type="password"] { padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 1em; }
        button, input[type="submit"] { padding: 12px; background: #3498db; color: white; border: none; border-radius: 4px; font-size: 1em; cursor: pointer; }
        button:hover, input[type="submit"]:hover { background: #2980b9; }
        .error { background: #e74c3c; color: white; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .success { background: #27ae60; color: white; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .info { background: #3498db; color: white; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .links { margin-top: 15px; }
        .links a { color: #3498db; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
        .welcome-box { text-align: center; }
        .welcome-box h1 { font-size: 2em; }
        .welcome-box p { color: #666; font-size: 1.1em; margin-bottom: 10px; }
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
    <div class="container">
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

@app.route('/')
def index():
    if session.get('user_id'):
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
        <h2>Register</h2>
        <form method="post">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required>
            <label for="password">Password</label>
            <input type="password" id="password" name="password" required>
            <label for="confirm_password">Confirm Password</label>
            <input type="password" id="confirm_password" name="confirm_password" required>
            <input type="submit" value="Register">
        </form>
        <div class="links">
            <p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
        </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Register', content=render_template_string(content), error=error, success=success, session=session)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            error = 'Please enter both username and password.'
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

    content = '''
        <h2>Login</h2>
        <form method="post">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required>
            <label for="password">Password</label>
            <input type="password" id="password" name="password" required>
            <input type="submit" value="Login">
        </form>
        <div class="links">
            <p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
        </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Login', content=render_template_string(content), error=error, success=None, session=session)

@app.route('/welcome')
def welcome():
    if not session.get('user_id'):
        return redirect(url_for('login'))

    username = session.get('username', 'User')
    admin_link = ''
    if session.get('is_admin'):
        admin_link = '<p><a href="{{ url_for(\'admin_panel\') }}">Go to Admin Panel</a></p>'

    content = '''
        <div class="welcome-box">
            <h1>Welcome, ''' + username + '''!</h1>
            <p>You are successfully logged into the Company Portal.</p>
            <p>This is your dashboard. More features coming soon.</p>
            ''' + admin_link + '''
        </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Welcome', content=render_template_string(content), error=None, success=None, session=session)

@app.route('/admin')
def admin_panel():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        content = '<h2>Access Denied</h2><p>You do not have permission to view this page.</p>'
        return render_template_string(BASE_TEMPLATE, title='Access Denied', content=content, error=None, success=None, session=session), 403

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()

    rows = ''
    for user in users:
        role = 'Admin' if user['is_admin'] else 'User'
        rows += '<tr><td>{}</td><td>{}</td><td>{}</td><td><a href="/admin/delete/{}">Delete</a></td></tr>'.format(
            user['id'], user['username'], role, user['id'])

    content = '''
        <h2>Admin Panel - All Users</h2>
        <div class="info">Total registered users: ''' + str(len(users)) + '''</div>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Username</th>
                    <th>Role</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                ''' + rows + '''
            </tbody>
        </table>
    '''
    return render_template_string(BASE_TEMPLATE, title='Admin Panel', content=content, error=None, success=None, session=session)

@app.route('/admin/delete/<int:user_id>')
def delete_user(user_id):
    if not session.get('user_id'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))

    if user_id == session.get('user_id'):
        return redirect(url_for('admin_panel'))

    conn = get_db()
    conn.execute("DELETE FROM users WHERE id = ? AND is_admin = 0", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_panel'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)