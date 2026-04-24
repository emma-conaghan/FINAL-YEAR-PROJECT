from flask import Flask, render_template_string, request, redirect, url_for, session
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
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Create default admin if not exists
    admin_pass = hashlib.md5('admin123'.encode()).hexdigest()
    c.execute('INSERT OR IGNORE INTO users (username, password, is_admin) VALUES (?, ?, ?)',
              ('admin', admin_pass, 1))
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
        .container { max-width: 500px; margin: 80px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h2 { text-align: center; color: #333; }
        input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button { width: 100%; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        .error { color: red; text-align: center; }
        .success { color: green; text-align: center; }
        .link { text-align: center; margin-top: 15px; }
        a { color: #4CAF50; text-decoration: none; }
        nav { background: #333; padding: 10px 20px; }
        nav a { color: white; margin-right: 15px; text-decoration: none; }
        .wide { max-width: 800px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
    </style>
</head>
<body>
{% if session.get('username') %}
<nav>
    <a href="{{ url_for('welcome') }}">Home</a>
    {% if session.get('is_admin') %}<a href="{{ url_for('admin') }}">Admin</a>{% endif %}
    <a href="{{ url_for('logout') }}">Logout</a>
    <span style="color:white; float:right;">Logged in as: {{ session['username'] }}</span>
</nav>
{% endif %}
{{ content }}
</body>
</html>
'''

LOGIN_FORM = '''
<div class="container">
    <h2>Company Portal Login</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    <div class="link"><a href="{{ url_for('register') }}">Don't have an account? Register</a></div>
</div>
'''

REGISTER_FORM = '''
<div class="container">
    <h2>Register</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    {% if success %}<p class="success">{{ success }}</p>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <input type="password" name="confirm_password" placeholder="Confirm Password" required>
        <button type="submit">Register</button>
    </form>
    <div class="link"><a href="{{ url_for('login') }}">Already have an account? Login</a></div>
</div>
'''

WELCOME_PAGE = '''
<div class="container">
    <h2>Welcome, {{ username }}!</h2>
    <p style="text-align:center; color:#555;">You are logged into the Company Internal Portal.</p>
    <p style="text-align:center; color:#555;">Use the navigation bar above to explore.</p>
    {% if is_admin %}
    <p style="text-align:center;"><a href="{{ url_for('admin') }}">Go to Admin Panel</a></p>
    {% endif %}
</div>
'''

ADMIN_PAGE = '''
<div class="container wide">
    <h2>Admin Panel - All Users</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Role</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
        </tr>
        {% endfor %}
    </table>
    <p style="text-align:center; margin-top:20px; color:#888;">Total users: {{ users|length }}</p>
</div>
'''

def render_page(content_template, **kwargs):
    from flask import render_template_string as rts
    content = rts(content_template, **kwargs)
    return rts(BASE_TEMPLATE, content=content, session=session)

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        hashed = hash_password(password)
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username=? AND password=?',
                            (username, hashed)).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_page(LOGIN_FORM, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
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
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                             (username, hashed))
                conn.commit()
                conn.close()
                success = 'Registration successful! You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
    return render_page(REGISTER_FORM, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_page(WELCOME_PAGE, username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return 'Access denied. Admins only.', 403
    conn = get_db()
    users = conn.execute('SELECT id, username, is_admin FROM users ORDER BY id').fetchall()
    conn.close()
    return render_page(ADMIN_PAGE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)