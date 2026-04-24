import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string

app = Flask(__name__)
app.secret_key = 'internal_secret_key'

def get_db_connection():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)')
    conn.commit()
    conn.close()

LAYOUT = """
<!DOCTYPE html>
<html>
<head><title>Company Portal</title></head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        {% if session.get('username') %}
            <a href="{{ url_for('admin') }}">Admin</a> | 
            <a href="{{ url_for('logout') }}">Logout ({{ session['username'] }})</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> | 
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    <hr>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p style="color: red;">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
"""

INDEX_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>Welcome to the Internal Portal</h1>
    {% if session.get('username') %}
        <p>Hello, {{ session['username'] }}! You are successfully logged in.</p>
    {% else %}
        <p>Please log in or register to access the portal.</p>
    {% endif %}
{% endblock %}
"""

LOGIN_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>Login</h1>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Login">
    </form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>Register</h1>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Register">
    </form>
{% endblock %}
"""

ADMIN_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>Admin Area - Registered Users</h1>
    <table border="1">
        <tr><th>ID</th><th>Username</th><th>Password (Plaintext)</th></tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ user['password'] }}</td>
        </tr>
        {% endfor %}
    </table>
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
        finally:
            conn.close()
    return render_template_string(REGISTER_HTML)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            return redirect(url_for('index'))
        return "Invalid credentials."
    return render_template_string(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/admin')
def admin():
    if not session.get('username'):
        return redirect(url_for('login'))
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return render_template_string(ADMIN_HTML, users=users)

@app.context_processor
def inject_layout():
    return dict(layout=LAYOUT)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)