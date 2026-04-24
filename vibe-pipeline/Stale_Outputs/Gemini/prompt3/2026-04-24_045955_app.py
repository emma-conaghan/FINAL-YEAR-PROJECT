import sqlite3
from flask import Flask, request, render_template_string, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'dev-secret-key-12345'

DATABASE = 'portal.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        ''')
        # Check if admin exists, if not create one: admin / admin123
        admin = conn.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()
        if not admin:
            hashed_pw = generate_password_hash('admin123')
            conn.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', ('admin', hashed_pw, 'admin'))
        conn.commit()

# --- Templates ---
BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Internal Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; padding: 10px; background: #f4f4f4; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        {% if session.get('user_id') %}
            Logged in as <strong>{{ session.get('username') }}</strong>
            {% if session.get('role') == 'admin' %}
                | <a href="{{ url_for('admin') }}">Admin Area</a>
            {% endif %}
            | <a href="{{ url_for('logout') }}">Logout</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> | 
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}<p class="error">{{ message }}</p>{% endfor %}
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
"""

LOGIN_HTML = """
{% extends "base" %}
{% block content %}
    <h2>Login</h2>
    <form method="post">
        <label>Username:</label><br>
        <input type="text" name="username" required><br>
        <label>Password:</label><br>
        <input type="password" name="password" required><br><br>
        <button type="submit">Login</button>
    </form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "base" %}
{% block content %}
    <h2>Register</h2>
    <form method="post">
        <label>Username:</label><br>
        <input type="text" name="username" required><br>
        <label>Password:</label><br>
        <input type="password" name="password" required><br><br>
        <button type="submit">Sign Up</button>
    </form>
{% endblock %}
"""

INDEX_HTML = """
{% extends "base" %}
{% block content %}
    <h1>Welcome to the Internal Portal</h1>
    <p>This is a secure area for company employees.</p>
    <p>You are authenticated and have access to private resources.</p>
{% endblock %}
"""

ADMIN_HTML = """
{% extends "base" %}
{% block content %}
    <h2>Administrator Panel</h2>
    <h3>Registered Users</h3>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Role</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user.id }}</td>
            <td>{{ user.username }}</td>
            <td>{{ user.role }}</td>
        </tr>
        {% endfor %}
    </table>
{% endblock %}
"""

# Helper to render string templates mimicking file inheritance
def render_portal_template(template_str, **context):
    # This effectively injects the base layout into the child templates
    full_template = template_str.replace('{% extends "base" %}', BASE_HTML)
    return render_template_string(full_template, **context)

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_portal_template(INDEX_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash("All fields are required.")
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
            conn.commit()
            conn.close()
            flash("Registration successful! Please log in.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
            
    return render_portal_template(REGISTER_HTML)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.")

    return render_portal_template(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if session.get('role') != 'admin':
        return "Access Denied", 403
    
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, role FROM users').fetchall()
    conn.close()
    return render_portal_template(ADMIN_HTML, users=users)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)