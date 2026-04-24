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
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Create default admin if not exists
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                       ('admin', admin_password, 1))
    conn.commit()
    conn.close()

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
    .container { max-width: 400px; margin: 80px auto; background: white; padding: 30px;
                 border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h2 { text-align: center; color: #333; }
    input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box;
            border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
    button { width: 100%; padding: 10px; background: #4CAF50; color: white;
             border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
    button:hover { background: #45a049; }
    .error { color: red; text-align: center; margin: 10px 0; }
    .success { color: green; text-align: center; margin: 10px 0; }
    a { color: #4CAF50; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .link-center { text-align: center; margin-top: 15px; }
    .nav { background: #333; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }
    .nav a { color: white; margin-left: 15px; }
    .nav span { color: white; }
    .main-content { max-width: 800px; margin: 40px auto; background: white;
                    padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 12px; border: 1px solid #ddd; text-align: left; }
    th { background: #4CAF50; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
    .badge { padding: 3px 8px; border-radius: 12px; font-size: 12px; }
    .badge-admin { background: #ff9800; color: white; }
    .badge-user { background: #2196F3; color: white; }
</style>
'''

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Login - Company Portal</title>''' + BASE_STYLE + '''</head>
<body>
<div class="container">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Sign In</h3>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    <div class="link-center">
        Don't have an account? <a href="/register">Register here</a>
    </div>
</div>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Register - Company Portal</title>''' + BASE_STYLE + '''</head>
<body>
<div class="container">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Create Account</h3>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    {% if success %}<p class="success">{{ success }}</p>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <input type="password" name="confirm_password" placeholder="Confirm Password" required>
        <button type="submit">Register</button>
    </form>
    <div class="link-center">
        Already have an account? <a href="/login">Login here</a>
    </div>
</div>
</body>
</html>
'''

WELCOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Welcome - Company Portal</title>''' + BASE_STYLE + '''</head>
<body>
<div class="nav">
    <span>🏢 Company Portal</span>
    <div>
        <span style="color:#aaa;">Hello, {{ username }}</span>
        {% if is_admin %}<a href="/admin">Admin Panel</a>{% endif %}
        <a href="/logout">Logout</a>
    </div>
</div>
<div class="main-content">
    <h2>Welcome, {{ username }}! 👋</h2>
    <p>You have successfully logged into the Company Internal Portal.</p>
    <div style="background:#e8f5e9; padding:20px; border-radius:8px; margin-top:20px;">
        <h3>📋 Portal Features</h3>
        <ul>
            <li>Access company resources and documents</li>
            <li>Connect with your colleagues</li>
            <li>View company announcements</li>
            <li>Manage your profile settings</li>
        </ul>
    </div>
    {% if is_admin %}
    <div style="background:#fff3e0; padding:20px; border-radius:8px; margin-top:20px;">
        <h3>⚙️ Admin Access</h3>
        <p>You have administrator privileges. <a href="/admin">Go to Admin Panel</a></p>
    </div>
    {% endif %}
</div>
</body>
</html>
'''

ADMIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Admin Panel - Company Portal</title>''' + BASE_STYLE + '''</head>
<body>
<div class="nav">
    <span>🏢 Company Portal - Admin Panel</span>
    <div>
        <a href="/welcome">Home</a>
        <a href="/logout">Logout</a>
    </div>
</div>
<div class="main-content">
    <h2>⚙️ Admin Panel</h2>
    <p>Manage all registered users in the system.</p>
    <p><strong>Total Users:</strong> {{ users|length }}</p>
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
                <td>
                    {% if user['is_admin'] %}
                    <span class="badge badge-admin">Admin</span>
                    {% else %}
                    <span class="badge badge-user">User</span>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
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
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed = hashlib.md5(password.encode()).hexdigest()

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed))
        user = cursor.fetchone()
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
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            error = 'Passwords do not match.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        else:
            hashed = hashlib.md5(password.encode()).hexdigest()
            conn = get_db()
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
                conn.commit()
                success = 'Account created successfully! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose a different one.'
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
    if not session.get('is_admin', False):
        return redirect(url_for('welcome'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, is_admin FROM users ORDER BY id')
    users = cursor.fetchall()
    conn.close()

    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)