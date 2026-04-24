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
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; flex-direction: column; }
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 50px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button, .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .alert { padding: 10px 15px; border-radius: 4px; margin-bottom: 15px; }
        .alert-error { background: #fce4e4; color: #c0392b; border: 1px solid #f5c6cb; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .links { margin-top: 15px; }
        .links a { color: #3498db; }
    </style>
</head>
<body>
    <div class="navbar">
        <strong>Company Portal</strong>
        <div>
            {% if session.get('username') %}
                <span>Hello, {{ session['username'] }}</span>
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
                error = 'Username already exists.'
            finally:
                conn.close()

    content = '''
        <h2>Register</h2>
        {% if error %}
            <div class="alert alert-error">{{ error }}</div>
        {% endif %}
        {% if success %}
            <div class="alert alert-success">{{ success }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required>
            </div>
            <div class="form-group">
                <label>Confirm Password</label>
                <input type="password" name="confirm_password" required>
            </div>
            <button type="submit">Register</button>
        </form>
        <div class="links">
            <p>Already have an account? <a href="/login">Login here</a></p>
        </div>
    '''

    full_template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(full_template, title='Register', error=error, success=success, session=session)

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
        <h2>Login</h2>
        {% if error %}
            <div class="alert alert-error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <div class="form-group">
                <label>Username</label>
                <input type="text" name="username" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required>
            </div>
            <button type="submit">Login</button>
        </form>
        <div class="links">
            <p>Don't have an account? <a href="/register">Register here</a></p>
        </div>
    '''

    full_template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(full_template, title='Login', error=error, session=session)

@app.route('/welcome')
def welcome():
    if not session.get('username'):
        return redirect(url_for('login'))

    content = '''
        <h2>Welcome, {{ session['username'] }}!</h2>
        <p style="margin-bottom: 20px; color: #555; line-height: 1.6;">
            You have successfully logged into the Company Portal.
            This is your personal dashboard where you can access company resources.
        </p>
        <div style="background: #ecf0f1; padding: 20px; border-radius: 4px; margin-bottom: 20px;">
            <h3 style="color: #2c3e50; margin-bottom: 10px;">Quick Info</h3>
            <p style="color: #555;">User ID: {{ session['user_id'] }}</p>
            <p style="color: #555;">Role: {{ 'Administrator' if session.get('is_admin') else 'Standard User' }}</p>
        </div>
        {% if session.get('is_admin') %}
            <a href="/admin" class="btn btn-success">Go to Admin Panel</a>
        {% endif %}
    '''

    full_template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(full_template, title='Welcome', session=session)

@app.route('/admin')
def admin():
    if not session.get('username'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        content = '''
            <h2>Access Denied</h2>
            <div class="alert alert-error">You do not have permission to access the admin area.</div>
            <a href="/welcome" class="btn">Back to Home</a>
        '''
        full_template = BASE_TEMPLATE.replace('{{ content }}', content)
        return render_template_string(full_template, title='Access Denied', session=session), 403

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()

    content = '''
        <h2>Admin Panel - Registered Users</h2>
        <p style="margin-bottom: 15px; color: #555;">Total users: {{ users|length }}</p>
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
                {% for user in users %}
                <tr>
                    <td>{{ user['id'] }}</td>
                    <td>{{ user['username'] }}</td>
                    <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
                    <td>
                        {% if not user['is_admin'] %}
                            <a href="/admin/delete/{{ user['id'] }}" class="btn btn-danger" 
                               onclick="return confirm('Are you sure you want to delete this user?');"
                               style="padding: 5px 10px; font-size: 12px;">Delete</a>
                        {% else %}
                            <span style="color: #999;">-</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    '''

    full_template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(full_template, title='Admin', users=users, session=session)

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
    return redirect(url_for('admin'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)