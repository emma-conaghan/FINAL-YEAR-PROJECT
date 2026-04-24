import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

def init_db():
    with sqlite3.connect('portal.db') as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         username TEXT UNIQUE, 
                         password TEXT, 
                         is_admin INTEGER)''')
        try:
            conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', 'admin123', 1))
        except sqlite3.IntegrityError:
            pass
        conn.commit()

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head><title>Company Portal</title></head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        {% if 'user_id' in session %}
            <a href="{{ url_for('logout') }}">Logout</a>
            {% if session.get('is_admin') %}
                | <a href="{{ url_for('admin') }}">Admin Area</a>
            {% endif %}
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> | <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    <hr>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul>{% for msg in messages %}<li>{{ msg }}</li>{% endfor %}</ul>
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
"""

INDEX_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>Welcome, {{ username }}!</h1>
    <p>This is the internal company portal dashboard.</p>
{% endblock %}
"""

LOGIN_HTML = """
{% extends "layout" %}
{% block content %}
    <h2>Login</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <button type="submit">Login</button>
    </form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "layout" %}
{% block content %}
    <h2>Register</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <button type="submit">Sign Up</button>
    </form>
{% endblock %}
"""

ADMIN_HTML = """
{% extends "layout" %}
{% block content %}
    <h2>Admin Area - Registered Users</h2>
    <table border="1">
        <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
        </tr>
        {% endfor %}
    </table>
{% endblock %}
"""

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(INDEX_HTML, username=session.get('username'), layout=HTML_LAYOUT)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            with sqlite3.connect('portal.db') as conn:
                conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", (username, password, 0))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
    return render_template_string(REGISTER_HTML, layout=HTML_LAYOUT)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect('portal.db') as conn:
            conn.row_factory = sqlite3.Row
            user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = user['is_admin']
                return redirect(url_for('index'))
            return "Invalid credentials"
    return render_template_string(LOGIN_HTML, layout=HTML_LAYOUT)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return "Access Denied", 403
    with sqlite3.connect('portal.db') as conn:
        conn.row_factory = sqlite3.Row
        users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
    return render_template_string(ADMIN_HTML, users=users, layout=HTML_LAYOUT)

@app.context_processor
def inject_layout():
    return dict(layout=HTML_LAYOUT)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)