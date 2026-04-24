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
        h1, h2 { color: #333; }
        input[type=text], input[type=password] { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button, .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #45a049; }
        .btn-red { background: #e53935; }
        .btn-red:hover { background: #c62828; }
        .btn-blue { background: #1976D2; }
        .btn-blue:hover { background: #1565C0; }
        .error { color: red; margin: 10px 0; }
        .success { color: green; margin: 10px 0; }
        .nav { background: #333; padding: 10px 20px; color: white; display: flex; justify-content: space-between; align-items: center; }
        .nav a { color: white; text-decoration: none; margin-left: 15px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4CAF50; color: white; }
        tr:hover { background: #f5f5f5; }
        .wide { max-width: 800px; }
    </style>
</head>
<body>
{% if session.get('username') %}
<div class="nav">
    <span>Company Internal Portal</span>
    <div>
        <span>Welcome, {{ session['username'] }}</span>
        {% if session.get('is_admin') %}<a href="/admin">Admin Panel</a>{% endif %}
        <a href="/logout">Logout</a>
    </div>
</div>
{% endif %}
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
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'

    html = BASE_TEMPLATE + '''
<div class="container">
    <h2>Login</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    <p>Don't have an account? <a href="/register">Register here</a></p>
</div>
</body></html>
'''
    return render_template_string(html, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    success = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')

        if not username or not password:
            error = 'Username and password are required.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        elif password != confirm:
            error = 'Passwords do not match.'
        else:
            hashed = hash_password(password)
            conn = get_db()
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
                conn.commit()
                success = 'Registration successful! You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            finally:
                conn.close()

    html = BASE_TEMPLATE + '''
<div class="container">
    <h2>Register</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    {% if success %}<p class="success">{{ success }}</p>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <input type="password" name="confirm" placeholder="Confirm Password" required>
        <button type="submit">Register</button>
    </form>
    <p>Already have an account? <a href="/login">Login here</a></p>
</div>
</body></html>
'''
    return render_template_string(html, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))

    html = BASE_TEMPLATE + '''
<div class="container">
    <h1>Welcome, {{ username }}!</h1>
    <p>You are logged into the Company Internal Portal.</p>
    <p>This is your personal dashboard. More features coming soon!</p>
    {% if is_admin %}
    <p><a class="btn btn-blue" href="/admin">Go to Admin Panel</a></p>
    {% endif %}
    <p><a class="btn btn-red" href="/logout">Logout</a></p>
</div>
</body></html>
'''
    return render_template_string(html,
                                  username=session['username'],
                                  is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, is_admin FROM users ORDER BY id')
    users = cursor.fetchall()
    conn.close()

    html = BASE_TEMPLATE + '''
<div class="container wide" style="max-width:800px; margin: 40px auto;">
    <h2>Admin Panel - All Users</h2>
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
            <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
            <td>
                {% if user['username'] != session['username'] %}
                <a class="btn btn-red" href="/admin/delete/{{ user['id'] }}" onclick="return confirm('Delete this user?')">Delete</a>
                {% else %}
                <em>Current user</em>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
    <br>
    <a class="btn" href="/welcome">Back to Welcome</a>
</div>
</body></html>
'''
    return render_template_string(html, users=users)

@app.route('/admin/delete/<int:user_id>')
def delete_user(user_id):
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    if user and user['username'] != session['username']:
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
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