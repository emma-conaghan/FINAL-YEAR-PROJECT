import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

DATABASE = 'portal.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         username TEXT UNIQUE NOT NULL, 
                         password TEXT NOT NULL, 
                         is_admin INTEGER DEFAULT 0)''')
        conn.commit()

init_db()

LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; background: #f4f4f4; }
        .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); max-width: 600px; margin: auto; }
        input { display: block; width: 100%; padding: 10px; margin: 10px 0; box-sizing: border-box; }
        button { background: #007bff; color: white; border: none; padding: 10px 15px; cursor: pointer; border-radius: 4px; }
        .nav { margin-bottom: 20px; }
        .nav a { margin-right: 15px; text-decoration: none; color: #007bff; }
        .error { color: red; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            {% if session.get('user_id') %}
                <a href="{{ url_for('dashboard') }}">Dashboard</a>
                {% if session.get('is_admin') %}
                    <a href="{{ url_for('admin') }}">Admin Panel</a>
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            {% for message in messages %}
              <p class="error">{{ message }}</p>
            {% endfor %}
          {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

LOGIN_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Login</h2>
<form method="post">
    <input type="text" name="username" placeholder="Username" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Login</button>
</form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Register</h2>
<form method="post">
    <input type="text" name="username" placeholder="Username" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Sign Up</button>
</form>
{% endblock %}
"""

DASHBOARD_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Welcome, {{ username }}!</h2>
<p>This is your internal company portal dashboard.</p>
{% endblock %}
"""

ADMIN_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Admin Panel - Registered Users</h2>
<table>
    <tr>
        <th>ID</th>
        <th>Username</th>
        <th>Is Admin</th>
    </tr>
    {% for user in users %}
    <tr>
        <td>{{ user.id }}</td>
        <td>{{ user.username }}</td>
        <td>{{ 'Yes' if user.is_admin else 'No' }}</td>
    </tr>
    {% endfor %}
</table>
{% endblock %}
"""

@app.context_processor
def inject_layout():
    return {'layout': LAYOUT}

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        
        db = get_db()
        try:
            # First user to register becomes admin
            cursor = db.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            is_admin = 1 if count == 0 else 0
            
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", 
                       (username, password, is_admin))
            db.commit()
            flash("Registration successful. Please login.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
    return render_template_string(REGISTER_HTML)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials.")
    return render_template_string(LOGIN_HTML)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(DASHBOARD_HTML, username=session['username'])

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        flash("Access denied.")
        return redirect(url_for('dashboard'))
    
    db = get_db()
    users = db.execute("SELECT id, username, is_admin FROM users").fetchall()
    return render_template_string(ADMIN_HTML, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)