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
    # Create default admin account
    try:
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     ('admin', 'admin123', 1))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f6f9; }
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
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
        tr:nth-child(even) { background: #f2f2f2; }
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
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Login</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
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
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    {% if success %}<p class="success">{{ success }}</p>{% endif %}
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

WELCOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Welcome, {{ username }}!</h2>
    <p>You are successfully logged into the Company Portal.</p>
    <p>This is the internal company portal. Use the navigation above to access different areas.</p>
    {% if is_admin %}
    <p><strong>You have administrator privileges.</strong> <a href="/admin">Go to Admin Panel</a></p>
    {% endif %}
''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Admin Panel - Registered Users</h2>
    <p>Total users: {{ users|length }}</p>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Password</th>
            <th>Role</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ user['password'] }}</td>
            <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
        </tr>
        {% endfor %}
    </table>
''')

HOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h2>Welcome to the Company Portal</h2>
    <p>Please <a href="/login">login</a> or <a href="/register">register</a> to continue.</p>
''')


@app.route('/')
def index():
    if session.get('username'):
        return redirect(url_for('welcome'))
    return render_template_string(HOME_TEMPLATE, session=session)


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
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            finally:
                conn.close()

    return render_template_string(REGISTER_TEMPLATE, error=error, success=success, session=session)


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
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

    return render_template_string(LOGIN_TEMPLATE, error=error, session=session)


@app.route('/welcome')
def welcome():
    if not session.get('username'):
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE,
                                  username=session['username'],
                                  is_admin=session.get('is_admin', False),
                                  session=session)


@app.route('/admin')
def admin():
    if not session.get('username'):
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return "Access denied. Admin privileges required.", 403

    conn = get_db()
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()

    return render_template_string(ADMIN_TEMPLATE, users=users, session=session)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)