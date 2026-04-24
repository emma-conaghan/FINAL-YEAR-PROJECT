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
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    c.execute('INSERT OR IGNORE INTO users (username, password, is_admin) VALUES (?, ?, ?)',
              ('admin', admin_password, 1))
    conn.commit()
    conn.close()

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
    .container { max-width: 500px; margin: 80px auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #333; text-align: center; margin-bottom: 30px; }
    h2 { color: #555; }
    input[type=text], input[type=password] { width: 100%; padding: 10px; margin: 8px 0 16px 0; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
    button, .btn { background: #4CAF50; color: white; padding: 12px 20px; border: none; border-radius: 5px; cursor: pointer; width: 100%; font-size: 16px; text-decoration: none; display: inline-block; text-align: center; box-sizing: border-box; }
    button:hover, .btn:hover { background: #45a049; }
    .btn-danger { background: #e74c3c; }
    .btn-danger:hover { background: #c0392b; }
    .btn-secondary { background: #3498db; }
    .btn-secondary:hover { background: #2980b9; }
    .error { color: red; margin-bottom: 15px; text-align: center; }
    .success { color: green; margin-bottom: 15px; text-align: center; }
    .nav { text-align: center; margin-top: 20px; }
    .nav a { color: #3498db; text-decoration: none; margin: 0 10px; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background-color: #4CAF50; color: white; }
    tr:hover { background-color: #f5f5f5; }
    .badge { padding: 3px 8px; border-radius: 3px; font-size: 12px; }
    .badge-admin { background: #e74c3c; color: white; }
    .badge-user { background: #3498db; color: white; }
    .wide-container { max-width: 800px; margin: 80px auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
</style>
'''

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Login - Company Portal</title>''' + BASE_STYLE + '''</head>
<body>
<div class="container">
    <h1>🏢 Company Portal</h1>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if success %}<div class="success">{{ success }}</div>{% endif %}
    <form method="POST">
        <label>Username</label>
        <input type="text" name="username" placeholder="Enter username" required>
        <label>Password</label>
        <input type="password" name="password" placeholder="Enter password" required>
        <button type="submit">Login</button>
    </form>
    <div class="nav">
        <a href="/register">Don't have an account? Register</a>
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
    <h1>📝 Register</h1>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST">
        <label>Username</label>
        <input type="text" name="username" placeholder="Choose a username" required>
        <label>Password</label>
        <input type="password" name="password" placeholder="Choose a password" required>
        <label>Confirm Password</label>
        <input type="password" name="confirm_password" placeholder="Confirm your password" required>
        <button type="submit">Register</button>
    </form>
    <div class="nav">
        <a href="/login">Already have an account? Login</a>
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
<div class="container">
    <h1>👋 Welcome!</h1>
    <p style="text-align:center; color:#555; font-size:18px;">Hello, <strong>{{ username }}</strong>!</p>
    <p style="text-align:center; color:#777;">You are successfully logged into the Company Portal.</p>
    <br>
    {% if is_admin %}
    <a href="/admin" class="btn btn-secondary" style="margin-bottom:10px;">🔧 Admin Panel</a>
    {% endif %}
    <a href="/logout" class="btn btn-danger">Logout</a>
</div>
</body>
</html>
'''

ADMIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Admin - Company Portal</title>''' + BASE_STYLE + '''</head>
<body>
<div class="wide-container">
    <h1>🔧 Admin Panel</h1>
    <h2>Registered Users ({{ users|length }})</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Role</th>
            <th>Action</th>
        </tr>
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
            <td>
                {% if user['username'] != session['username'] %}
                <a href="/admin/delete/{{ user['id'] }}" onclick="return confirm('Delete this user?')" style="color:red; text-decoration:none;">Delete</a>
                {% else %}
                <span style="color:#999;">Current User</span>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
    <br>
    <a href="/welcome" class="btn btn-secondary">⬅ Back to Welcome</a>
    <br><br>
    <a href="/logout" class="btn btn-danger">Logout</a>
</div>
</body>
</html>
'''

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        hashed = hashlib.md5(password.encode()).hexdigest()
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
    if request.args.get('registered'):
        success = 'Registration successful! Please login.'
    return render_template_string(LOGIN_TEMPLATE, error=error, success=success)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
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
            try:
                conn = get_db()
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                             (username, hashed))
                conn.commit()
                conn.close()
                return redirect(url_for('login', registered=1))
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
    return render_template_string(REGISTER_TEMPLATE, error=error)

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
    users = conn.execute('SELECT * FROM users ORDER BY id').fetchall()
    conn.close()
    return render_template_string(ADMIN_TEMPLATE, users=users, session=session)

@app.route('/admin/delete/<int:user_id>')
def delete_user(user_id):
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    conn = get_db()
    user = conn.execute('SELECT username FROM users WHERE id=?', (user_id,)).fetchone()
    if user and user['username'] != session['username']:
        conn.execute('DELETE FROM users WHERE id=?', (user_id,))
        conn.commit()
    conn.close()
    return redirect(url_for('admin'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)