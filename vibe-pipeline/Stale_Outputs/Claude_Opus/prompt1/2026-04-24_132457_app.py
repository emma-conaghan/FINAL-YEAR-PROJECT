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
        nav .brand { font-size: 18px; font-weight: bold; color: white; }
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
        .info { background: #eef; color: #2980b9; padding: 10px; border-radius: 4px; margin-bottom: 15px; border: 1px solid #ccf; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .welcome-box { text-align: center; padding: 20px; }
        .welcome-box h1 { font-size: 28px; margin-bottom: 10px; }
        .welcome-box p { color: #666; font-size: 16px; margin-bottom: 20px; }
        .links { margin-top: 10px; }
        .links a { margin-right: 10px; }
    </style>
</head>
<body>
    <nav>
        <span class="brand">Company Portal</span>
        <div>
            {% if session.get('username') %}
                <a href="{{ url_for('welcome') }}">Home</a>
                {% if session.get('is_admin') %}
                    <a href="{{ url_for('admin_panel') }}">Admin</a>
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout ({{ session['username'] }})</a>
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

@app.route('/')
def index():
    if session.get('username'):
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
            <input type="text" id="username" name="username" placeholder="Choose a username" required>
            <label for="password">Password</label>
            <input type="password" id="password" name="password" placeholder="Choose a password" required>
            <label for="confirm_password">Confirm Password</label>
            <input type="password" id="confirm_password" name="confirm_password" placeholder="Confirm your password" required>
            <button type="submit">Register</button>
        </form>
        <div class="links">
            <p style="margin-top:15px; color:#666;">Already have an account? <a href="''' + url_for('login') + '''">Login here</a></p>
        </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Register', content=content,
                                  error=error, success=success, session=session)

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
                session['username'] = user['username']
                session['user_id'] = user['id']
                session['is_admin'] = bool(user['is_admin'])
                return redirect(url_for('welcome'))
            else:
                error = 'Invalid username or password.'

    content = '''
        <h2>Login</h2>
        <form method="post">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" placeholder="Enter your username" required>
            <label for="password">Password</label>
            <input type="password" id="password" name="password" placeholder="Enter your password" required>
            <button type="submit">Login</button>
        </form>
        <div class="links">
            <p style="margin-top:15px; color:#666;">Don't have an account? <a href="''' + url_for('register') + '''">Register here</a></p>
        </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Login', content=content,
                                  error=error, success=None, session=session)

@app.route('/welcome')
def welcome():
    if not session.get('username'):
        return redirect(url_for('login'))

    admin_link = ''
    if session.get('is_admin'):
        admin_link = '<a class="btn" href="' + url_for('admin_panel') + '">Go to Admin Panel</a>'

    content = '''
        <div class="welcome-box">
            <h1>Welcome, ''' + session['username'] + '''!</h1>
            <p>You are successfully logged into the Company Portal.</p>
            <div class="info">You are logged in as <strong>''' + session['username'] + '''</strong>''' + (' (Administrator)' if session.get('is_admin') else ' (User)') + '''</div>
            ''' + admin_link + '''
        </div>
    '''
    return render_template_string(BASE_TEMPLATE, title='Welcome', content=content,
                                  error=None, success=None, session=session)

@app.route('/admin')
def admin_panel():
    if not session.get('username'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        content = '''
            <h2>Access Denied</h2>
            <p style="color:#e74c3c;">You do not have permission to access the admin area.</p>
            <a class="btn" href="''' + url_for('welcome') + '''">Back to Home</a>
        '''
        return render_template_string(BASE_TEMPLATE, title='Access Denied', content=content,
                                      error=None, success=None, session=session), 403

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()

    rows = ''
    for user in users:
        role = 'Admin' if user['is_admin'] else 'User'
        delete_btn = ''
        if not user['is_admin']:
            delete_btn = '<a class="btn btn-danger" href="' + url_for('delete_user', user_id=user['id']) + '" onclick="return confirm(\'Are you sure you want to delete this user?\')">Delete</a>'
        rows += '<tr><td>' + str(user['id']) + '</td><td>' + user['username'] + '</td><td>' + role + '</td><td>' + delete_btn + '</td></tr>'

    content = '''
        <h2>Admin Panel - Registered Users</h2>
        <p style="color:#666; margin-bottom:10px;">Total users: ''' + str(len(users)) + '''</p>
        <table>
            <tr><th>ID</th><th>Username</th><th>Role</th><th>Actions</th></tr>
            ''' + rows + '''
        </table>
    '''
    return render_template_string(BASE_TEMPLATE, title='Admin Panel', content=content,
                                  error=None, success=None, session=session,
                                  container_class='wide')

@app.route('/admin/delete/<int:user_id>')
def delete_user(user_id):
    if not session.get('username') or not session.get('is_admin'):
        return redirect(url_for('login'))

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user and not user['is_admin']:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
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