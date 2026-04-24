from flask import Flask, request, redirect, url_for, session, render_template_string
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DB_PATH = 'users.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    try:
        conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                     ('admin', admin_password, 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; background: #f0f2f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
    .card { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 350px; }
    h2 { text-align: center; color: #333; margin-bottom: 20px; }
    input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
    button { width: 100%; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 10px; }
    button:hover { background: #45a049; }
    .error { color: red; text-align: center; margin: 10px 0; }
    .success { color: green; text-align: center; margin: 10px 0; }
    a { color: #4CAF50; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .link-text { text-align: center; margin-top: 15px; font-size: 14px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #4CAF50; color: white; }
    tr:hover { background: #f5f5f5; }
    .nav { margin-bottom: 20px; }
    .btn-red { background: #e53935; }
    .btn-red:hover { background: #c62828; }
    .welcome-card { width: 500px; }
</style>
'''

LOGIN_TEMPLATE = BASE_STYLE + '''
<div class="card">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Login</h3>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    <div class="link-text">Don't have an account? <a href="/register">Register here</a></div>
</div>
'''

REGISTER_TEMPLATE = BASE_STYLE + '''
<div class="card">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Register</h3>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if success %}<div class="success">{{ success }}</div>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <input type="password" name="confirm_password" placeholder="Confirm Password" required>
        <button type="submit">Register</button>
    </form>
    <div class="link-text">Already have an account? <a href="/login">Login here</a></div>
</div>
'''

WELCOME_TEMPLATE = BASE_STYLE + '''
<div class="card welcome-card">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Welcome, {{ username }}!</h3>
    <p style="text-align:center; color:#888;">You are successfully logged into the internal company portal.</p>
    <div class="nav" style="text-align:center;">
        {% if is_admin %}
        <a href="/admin"><button style="width:auto; padding:8px 20px; margin-right:10px;">Admin Panel</button></a>
        {% endif %}
        <a href="/logout"><button class="btn-red" style="width:auto; padding:8px 20px;">Logout</button></a>
    </div>
</div>
'''

ADMIN_TEMPLATE = BASE_STYLE + '''
<div class="card" style="width:700px;">
    <h2>🏢 Admin Panel</h2>
    <div class="nav">
        <a href="/welcome"><button style="width:auto; padding:8px 20px; margin-right:10px;">← Back</button></a>
        <a href="/logout"><button class="btn-red" style="width:auto; padding:8px 20px;">Logout</button></a>
    </div>
    <h3 style="color:#666;">Registered Users ({{ users|length }})</h3>
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
</div>
'''

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        hashed = hashlib.md5(password.encode()).hexdigest()
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?',
                            (username, hashed)).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_template_string(LOGIN_TEMPLATE, error=error)

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
            error = 'Username must be at least 3 characters.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        else:
            hashed = hashlib.md5(password.encode()).hexdigest()
            conn = get_db()
            try:
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                             (username, hashed))
                conn.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            finally:
                conn.close()
    return render_template_string(REGISTER_TEMPLATE, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE,
                                  username=session['username'],
                                  is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))
    conn = get_db()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)