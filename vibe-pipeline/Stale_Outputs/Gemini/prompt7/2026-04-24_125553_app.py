import sqlite3
from flask import Flask, request, render_template_string, redirect, session, url_for

app = Flask(__name__)
app.secret_key = 'insecure_secret_key'

DATABASE = 'database.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>User App</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        .nav { margin-bottom: 20px; }
        .form-group { margin-bottom: 10px; }
        label { display: inline-block; width: 100px; }
        .profile-box { border: 1px solid #ccc; padding: 15px; width: 300px; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="{{ url_for('index') }}">Home</a> |
        {% if session.get('user_id') %}
            <a href="{{ url_for('profile') }}">My Profile</a> |
            <a href="{{ url_for('logout') }}">Logout</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> |
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </div>
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
    <h1>Welcome to the User Portal</h1>
    <p>Find a user by ID:</p>
    <form action="/view_redirect" method="post">
        <input type="number" name="user_id" placeholder="User ID" required>
        <button type="submit">View Profile</button>
    </form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>Register</h1>
    <form method="post">
        <div class="form-group"><label>Username:</label><input type="text" name="username" required></div>
        <div class="form-group"><label>Password:</label><input type="password" name="password" required></div>
        <button type="submit">Create Account</button>
    </form>
{% endblock %}
"""

LOGIN_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>Login</h1>
    <form method="post">
        <div class="form-group"><label>Username:</label><input type="text" name="username" required></div>
        <div class="form-group"><label>Password:</label><input type="password" name="password" required></div>
        <button type="submit">Login</button>
    </form>
{% endblock %}
"""

PROFILE_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>Update Your Profile (ID: {{ user['id'] }})</h1>
    <form method="post">
        <div class="form-group"><label>Name:</label><input type="text" name="name" value="{{ user['name'] or '' }}"></div>
        <div class="form-group"><label>Email:</label><input type="email" name="email" value="{{ user['email'] or '' }}"></div>
        <div class="form-group"><label>Phone:</label><input type="text" name="phone" value="{{ user['phone'] or '' }}"></div>
        <div class="form-group"><label>Address:</label><input type="text" name="address" value="{{ user['address'] or '' }}"></div>
        <button type="submit">Update Information</button>
    </form>
{% endblock %}
"""

VIEW_HTML = """
{% extends "layout" %}
{% block content %}
    <h1>User Profile</h1>
    {% if user %}
        <div class="profile-box">
            <p><strong>ID:</strong> {{ user['id'] }}</p>
            <p><strong>Username:</strong> {{ user['username'] }}</p>
            <p><strong>Name:</strong> {{ user['name'] or 'N/A' }}</p>
            <p><strong>Email:</strong> {{ user['email'] or 'N/A' }}</p>
            <p><strong>Phone:</strong> {{ user['phone'] or 'N/A' }}</p>
            <p><strong>Address:</strong> {{ user['address'] or 'N/A' }}</p>
        </div>
    {% else %}
        <p>User not found.</p>
    {% endif %}
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/layout')
def layout():
    return HTML_LAYOUT

@app.context_processor
def inject_layout():
    return dict(layout=HTML_LAYOUT)

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
            session['user_id'] = user['id']
            return redirect(url_for('profile'))
        return "Invalid credentials."
    return render_template_string(LOGIN_HTML)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        conn.execute('UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?', 
                     (name, email, phone, address, session['user_id']))
        conn.commit()
        
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    return render_template_string(PROFILE_HTML, user=user)

@app.route('/user/<int:user_id>')
def view_user(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT id, username, name, email, phone, address FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return render_template_string(VIEW_HTML, user=user)

@app.route('/view_redirect', methods=['POST'])
def view_redirect():
    user_id = request.form.get('user_id')
    return redirect(url_for('view_user', user_id=user_id))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)