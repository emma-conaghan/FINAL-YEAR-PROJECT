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
    try:
        cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                      ('admin', admin_password, 1))
    except sqlite3.IntegrityError:
        pass
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
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        input { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { background: #4CAF50; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
        button:hover { background: #45a049; }
        .error { color: red; margin: 10px 0; }
        .success { color: green; margin: 10px 0; }
        a { color: #4CAF50; text-decoration: none; }
        a:hover { text-decoration: underline; }
        nav { margin-bottom: 20px; }
        nav a { margin-right: 15px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; border: 1px solid #ddd; text-align: left; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
    </style>
</head>
<body>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h1>🏢 Company Portal Login</h1>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
<form method="POST">
    <input type="text" name="username" placeholder="Username" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="/register">Register here</a></p>
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h1>🏢 Register</h1>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if success %}<p class="success">{{ success }}</p>{% endif %}
<form method="POST">
    <input type="text" name="username" placeholder="Choose a username" required>
    <input type="password" name="password" placeholder="Choose a password" required>
    <input type="password" name="confirm_password" placeholder="Confirm password" required>
    <button type="submit">Register</button>
</form>
<p>Already have an account? <a href="/">Login here</a></p>
''')

WELCOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<nav>
    {% if is_admin %}<a href="/admin">Admin Panel</a>{% endif %}
    <a href="/logout">Logout</a>
</nav>
<h1>👋 Welcome, {{ username }}!</h1>
<p>You are logged in to the Company Internal Portal.</p>
<p>This is your personal dashboard. More features coming soon!</p>
{% if is_admin %}<p><strong>You have administrator privileges.</strong></p>{% endif %}
''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<nav>
    <a href="/welcome">Dashboard</a>
    <a href="/logout">Logout</a>
</nav>
<h1>🔧 Admin Panel</h1>
<h2>Registered Users</h2>
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
<p>Total users: {{ users|length }}</p>
''')

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        hashed = hash_password(password)
        
        conn = get_db()
        # Intentionally vulnerable to SQL injection as per insecure request
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashed}'"
        try:
            user = conn.execute(query).fetchone()
        except Exception as e:
            user = None
        conn.close()
        
        if user:
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
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
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        else:
            hashed = hash_password(password)
            conn = get_db()
            try:
                conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, 0)',
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
                                  is_admin=session.get('is_admin', 0))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))
    
    conn = get_db()
    users = conn.execute('SELECT id, username, is_admin FROM users').fetchall()
    conn.close()
    
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)