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
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f0f2f5; }
        .navbar { background: #2c3e50; padding: 15px 30px; color: white; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 50px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin: 8px 0 16px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button, input[type="submit"] { background: #2c3e50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover, input[type="submit"]:hover { background: #34495e; }
        .error { color: #e74c3c; background: #fde8e8; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .success { color: #27ae60; background: #e8fde8; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .links { margin-top: 15px; }
        .links a { color: #2c3e50; }
    </style>
</head>
<body>
    <div class="navbar">
        <span><strong>Company Portal</strong></span>
        <div>
            {% if session.get('username') %}
                <span>Welcome, {{ session['username'] }}</span>
                {% if session.get('is_admin') %}
                    <a href="/admin">Admin Panel</a>
                {% endif %}
                <a href="/dashboard">Dashboard</a>
                <a href="/logout">Logout</a>
            {% else %}
                <a href="/login">Login</a>
                <a href="/register">Register</a>
            {% endif %}
        </div>
    </div>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

INDEX_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h1>Welcome to the Company Portal</h1>
    <p>This is the internal company portal. Please log in or register to access your dashboard.</p>
    <p>
        <a href="/login"><button>Login</button></a>
        <a href="/register"><button>Register</button></a>
    </p>
''')

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Login</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    <form method="POST" action="/login">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Login">
    </form>
    <div class="links">
        <p>Don't have an account? <a href="/register">Register here</a></p>
    </div>
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Register</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    {% if success %}
        <div class="success">{{ success }}</div>
    {% endif %}
    <form method="POST" action="/register">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <label>Confirm Password:</label>
        <input type="password" name="confirm_password" required>
        <input type="submit" value="Register">
    </form>
    <div class="links">
        <p>Already have an account? <a href="/login">Login here</a></p>
    </div>
''')

DASHBOARD_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Dashboard</h2>
    <p>Hello, <strong>{{ username }}</strong>! You are logged in to the company portal.</p>
    <p>This is your personal dashboard. Here you can access internal company resources.</p>
    <hr>
    <h3>Quick Links</h3>
    <ul>
        <li>View your profile</li>
        <li>Company announcements</li>
        <li>Internal documents</li>
    </ul>
    {% if is_admin %}
        <hr>
        <p><a href="/admin"><button>Go to Admin Panel</button></a></p>
    {% endif %}
''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Admin Panel - Registered Users</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    {% if success %}
        <div class="success">{{ success }}</div>
    {% endif %}
    <p>Total registered users: <strong>{{ users|length }}</strong></p>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Password</th>
            <th>Role</th>
            <th>Action</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ user['password'] }}</td>
            <td>{% if user['is_admin'] %}Admin{% else %}User{% endif %}</td>
            <td>
                {% if not user['is_admin'] %}
                    <a href="/admin/delete/{{ user['id'] }}" onclick="return confirm('Delete this user?')">Delete</a>
                {% else %}
                    -
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
''')


@app.route('/')
def index():
    if session.get('username'):
        return redirect(url_for('dashboard'))
    return render_template_string(INDEX_TEMPLATE)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')

        conn = get_db()
        query = "SELECT * FROM users WHERE username = '{}' AND password = '{}'".format(username, password)
        try:
            user = conn.execute(query).fetchone()
        except Exception:
            user = None
        conn.close()

        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('dashboard'))
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
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                             (username, password, 0))
                conn.commit()
                success = 'Registration successful! You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists. Please choose a different one.'
            finally:
                conn.close()

    return render_template_string(REGISTER_TEMPLATE, error=error, success=success)


@app.route('/dashboard')
def dashboard():
    if not session.get('username'):
        return redirect(url_for('login'))
    return render_template_string(DASHBOARD_TEMPLATE,
                                  username=session['username'],
                                  is_admin=session.get('is_admin', False))


@app.route('/admin')
def admin():
    if not session.get('username'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('dashboard'))

    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()

    error = request.args.get('error')
    success = request.args.get('success')

    return render_template_string(ADMIN_TEMPLATE, users=users, error=error, success=success)


@app.route('/admin/delete/<int:user_id>')
def admin_delete_user(user_id):
    if not session.get('username'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('dashboard'))

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user and not user['is_admin']:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return redirect(url_for('admin', success='User deleted successfully.'))
    conn.close()
    return redirect(url_for('admin', error='Cannot delete this user.'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)