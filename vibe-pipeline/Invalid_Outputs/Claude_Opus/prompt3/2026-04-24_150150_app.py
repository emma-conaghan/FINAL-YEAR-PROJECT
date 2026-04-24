import sqlite3
import os
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DB_PATH = 'portal.db'


def get_db():
    conn = sqlite3.connect(DB_PATH)
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
                     ('admin', 'admin', 1))
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
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin: 8px 0 16px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button, input[type="submit"] { background: #2c3e50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover, input[type="submit"]:hover { background: #34495e; }
        .error { color: red; margin-bottom: 10px; }
        .success { color: green; margin-bottom: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #2c3e50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .links { margin-top: 15px; }
        .links a { color: #2c3e50; }
    </style>
</head>
<body>
    <div class="navbar">
        <span><strong>Company Portal</strong></span>
        <div>
            {% if session.get('username') %}
                <span>Hello, {{ session['username'] }}</span>
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
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

INDEX_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <h1>Welcome to the Company Portal</h1>
    {% if session.get('username') %}
        <p>You are logged in as <strong>{{ session['username'] }}</strong>.</p>
        <p>Welcome to our internal company portal. Use the navigation above to access your resources.</p>
        {% if session.get('is_admin') %}
            <p><a href="/admin">Go to Admin Panel</a></p>
        {% endif %}
    {% else %}
        <p>Please <a href="/login">log in</a> or <a href="/register">register</a> to access the portal.</p>
    {% endif %}
{% endblock %}
'''

LOGIN_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <h2>Login</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    {% if success %}
        <div class="success">{{ success }}</div>
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
{% endblock %}
'''

REGISTER_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <h2>Register</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
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
{% endblock %}
'''

WELCOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <h1>Welcome, {{ session['username'] }}!</h1>
    <p>You have successfully logged in to the Company Portal.</p>
    <p>This is your dashboard. Here you can access internal company resources and tools.</p>
    {% if session.get('is_admin') %}
        <p><strong>You have administrator privileges.</strong></p>
        <p><a href="/admin">Go to Admin Panel</a></p>
    {% endif %}
{% endblock %}
'''

ADMIN_TEMPLATE = '''
{% extends "base" %}
{% block content %}
    <h2>Admin Panel - Registered Users</h2>
    <p>Total registered users: <strong>{{ users|length }}</strong></p>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Password</th>
            <th>Admin</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ user['password'] }}</td>
            <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
        </tr>
        {% endfor %}
    </table>
{% endblock %}
'''


@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE, session=session, base=BASE_TEMPLATE)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not password:
            error = 'Username and password are required.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif len(password) < 3:
            error = 'Password must be at least 3 characters.'
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login', registered='1'))
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
                conn.close()

    return render_template_string(REGISTER_TEMPLATE, error=error, session=session, base=BASE_TEMPLATE)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    success = None

    if request.args.get('registered'):
        success = 'Registration successful! Please log in.'

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

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

    return render_template_string(LOGIN_TEMPLATE, error=error, success=success, session=session, base=BASE_TEMPLATE)


@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, session=session, base=BASE_TEMPLATE)


@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return 'Access denied. You are not an administrator.', 403

    conn = get_db()
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    return render_template_string(ADMIN_TEMPLATE, users=users, session=session, base=BASE_TEMPLATE)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# Override Jinja2 to support our base template pattern
original_render = render_template_string

@app.before_request
def setup_jinja():
    app.jinja_env.globals['base'] = BASE_TEMPLATE


# We need to handle the extends properly - let's use a different approach
# Replace the template system to work without files

INDEX_TEMPLATE_FULL = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h1>Welcome to the Company Portal</h1>
    {% if session.get('username') %}
        <p>You are logged in as <strong>{{ session['username'] }}</strong>.</p>
        <p>Welcome to our internal company portal. Use the navigation above to access your resources.</p>
        {% if session.get('is_admin') %}
            <p><a href="/admin">Go to Admin Panel</a></p>
        {% endif %}
    {% else %}
        <p>Please <a href="/login">log in</a> or <a href="/register">register</a> to access the portal.</p>
    {% endif %}
''')

LOGIN_TEMPLATE_FULL = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Login</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    {% if success %}
        <div class="success">{{ success }}</div>
    {% endif %}
    <form method="POST" action="/login">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Login">
    </form>
    <div class="links">
        <p>Don\'t have an account? <a href="/register">Register here</a></p>
    </div>
''')

REGISTER_TEMPLATE_FULL = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Register</h2>
    {% if error %}
        <div class="error">{{ error }}</div>
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

WELCOME_TEMPLATE_FULL = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h1>Welcome, {{ session['username'] }}!</h1>
    <p>You have successfully logged in to the Company Portal.</p>
    <p>This is your dashboard. Here you can access internal company resources and tools.</p>
    {% if session.get('is_admin') %}
        <p><strong>You have administrator privileges.</strong></p>
        <p><a href="/admin">Go to Admin Panel</a></p>
    {% endif %}
''')

ADMIN_TEMPLATE_FULL = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Admin Panel - Registered Users</h2>
    <p>Total registered users: <strong>{{ users|length }}</strong></p>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Password</th>
            <th>Admin</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ user['password'] }}</td>
            <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
        </tr>
        {% endfor %}
    </table>
''')


@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE_FULL, session=session)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not password:
            error = 'Username and password are required.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif len(password) < 3:
            error = 'Password must be at least 3 characters.'
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login', registered='1'))
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
                conn.close()

    return render_template_string(REGISTER_TEMPLATE_FULL, error=error, session=