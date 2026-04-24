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
    # Create default admin user
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password, is_admin)
        VALUES (?, ?, ?)
    ''', ('admin', admin_password, 1))
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
        body { font-family: Arial, sans-serif; background: #f0f0f0; margin: 0; padding: 0; }
        .container { max-width: 600px; margin: 60px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        a { color: #4CAF50; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .error { color: red; margin: 10px 0; }
        .success { color: green; margin: 10px 0; }
        .nav { background: #333; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }
        .nav a { color: white; margin-left: 15px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
    </style>
</head>
<body>
{% if session.get('username') %}
<div class="nav">
    <span style="color:white; font-weight:bold;">Company Portal</span>
    <div>
        <span style="color:#ccc;">Hello, {{ session['username'] }}</span>
        {% if session.get('is_admin') %}<a href="/admin">Admin</a>{% endif %}
        <a href="/logout">Logout</a>
    </div>
</div>
{% endif %}
<div class="container">
    {% block content %}{% endblock %}
</div>
</body>
</html>
'''

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h1>Login</h1>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<form method="POST" action="/login">
    <input type="text" name="username" placeholder="Username" required />
    <input type="password" name="password" placeholder="Password" required />
    <button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="/register">Register here</a></p>
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h1>Register</h1>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if success %}<p class="success">{{ success }}</p>{% endif %}
<form method="POST" action="/register">
    <input type="text" name="username" placeholder="Username" required />
    <input type="password" name="password" placeholder="Password" required />
    <input type="password" name="confirm_password" placeholder="Confirm Password" required />
    <button type="submit">Register</button>
</form>
<p>Already have an account? <a href="/login">Login here</a></p>
''')

WELCOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h1>Welcome, {{ username }}!</h1>
<p>You are successfully logged into the Company Portal.</p>
<p>This is your personal dashboard. More features coming soon!</p>
{% if is_admin %}
<p><a href="/admin">Go to Admin Area</a></p>
{% endif %}
''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h1>Admin Area - All Users</h1>
<p>Total users registered: <strong>{{ users|length }}</strong></p>
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
<br>
<a href="/welcome">Back to Welcome Page</a>
''')

@app.route('/')
def index():
    if session.get('username'):
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        hashed = hash_password(password)
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed)).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            return render_template_string(LOGIN_TEMPLATE, error='Invalid username or password', session=session)
    return render_template_string(LOGIN_TEMPLATE, error=None, session=session)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not username or not password:
            return render_template_string(REGISTER_TEMPLATE, error='Username and password are required.', success=None, session=session)
        if password != confirm_password:
            return render_template_string(REGISTER_TEMPLATE, error='Passwords do not match.', success=None, session=session)
        if len(password) < 4:
            return render_template_string(REGISTER_TEMPLATE, error='Password must be at least 4 characters.', success=None, session=session)
        hashed = hash_password(password)
        conn = get_db()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
            conn.commit()
            conn.close()
            return render_template_string(REGISTER_TEMPLATE, error=None, success='Registration successful! You can now login.', session=session)
        except sqlite3.IntegrityError:
            conn.close()
            return render_template_string(REGISTER_TEMPLATE, error='Username already exists.', success=None, session=session)
    return render_template_string(REGISTER_TEMPLATE, error=None, success=None, session=session)

@app.route('/welcome')
def welcome():
    if not session.get('username'):
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], is_admin=session.get('is_admin', False), session=session)

@app.route('/admin')
def admin():
    if not session.get('username'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return '<h2>Access Denied. Admins only.</h2><a href="/welcome">Go back</a>'
    conn = get_db()
    users = conn.execute('SELECT id, username, is_admin FROM users ORDER BY id').fetchall()
    conn.close()
    return render_template_string(ADMIN_TEMPLATE, users=users, session=session)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)