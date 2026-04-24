import sqlite3
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'super_secret_key'

def get_db_connection():
    conn = sqlite3.connect('app.db')
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

init_db()

BASE_LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>User Profile System</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .container { max-width: 600px; margin: auto; }
        input { display: block; width: 100%; margin-bottom: 10px; padding: 8px; }
        label { font-weight: bold; }
        nav { margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        .btn { padding: 10px 15px; background: #007bff; color: white; border: none; cursor: pointer; text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <a href="/">Home</a> | 
            {% if session.get('user_id') %}
                <a href="/profile/update">Update Profile</a> | 
                <a href="/logout">Logout</a>
            {% else %}
                <a href="/register">Register</a> | 
                <a href="/login">Login</a>
            {% endif %}
        </nav>
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

INDEX_HTML = """
{% extends "base" %}
{% block content %}
    <h1>User Portal</h1>
    <p>Welcome to the user management system.</p>
    <form action="/view_lookup" method="POST">
        <label>Find Profile by Account ID:</label>
        <input type="number" name="account_id" required>
        <button type="submit" class="btn">View Profile</button>
    </form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "base" %}
{% block content %}
    <h1>Create Account</h1>
    <form method="POST">
        <label>Username:</label><input type="text" name="username" required>
        <label>Password:</label><input type="password" name="password" required>
        <label>Full Name:</label><input type="text" name="name">
        <label>Email:</label><input type="email" name="email">
        <label>Phone:</label><input type="text" name="phone">
        <label>Address:</label><input type="text" name="address">
        <button type="submit" class="btn">Register</button>
    </form>
{% endblock %}
"""

LOGIN_HTML = """
{% extends "base" %}
{% block content %}
    <h1>Login</h1>
    <form method="POST">
        <label>Username:</label><input type="text" name="username" required>
        <label>Password:</label><input type="password" name="password" required>
        <button type="submit" class="btn">Login</button>
    </form>
{% endblock %}
"""

UPDATE_HTML = """
{% extends "base" %}
{% block content %}
    <h1>Update Profile</h1>
    <form method="POST">
        <label>Full Name:</label><input type="text" name="name" value="{{ user.name or '' }}">
        <label>Email:</label><input type="email" name="email" value="{{ user.email or '' }}">
        <label>Phone:</label><input type="text" name="phone" value="{{ user.phone or '' }}">
        <label>Address:</label><input type="text" name="address" value="{{ user.address or '' }}">
        <button type="submit" class="btn">Save Changes</button>
    </form>
{% endblock %}
"""

VIEW_HTML = """
{% extends "base" %}
{% block content %}
    <h1>Profile Details</h1>
    {% if user %}
        <p><strong>Account ID:</strong> {{ user.id }}</p>
        <p><strong>Name:</strong> {{ user.name }}</p>
        <p><strong>Email:</strong> {{ user.email }}</p>
        <p><strong>Phone:</strong> {{ user.phone }}</p>
        <p><strong>Address:</strong> {{ user.address }}</p>
    {% else %}
        <p>User not found.</p>
    {% endif %}
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, base=BASE_LAYOUT)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                         (username, password, name, email, phone, address))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
        finally:
            conn.close()
    return render_template_string(REGISTER_HTML, base=BASE_LAYOUT)

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
            return redirect(url_for('index'))
        return "Invalid credentials."
    return render_template_string(LOGIN_HTML, base=BASE_LAYOUT)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/profile/update', methods=['GET', 'POST'])
def update_profile():
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
        conn.close()
        return redirect(url_for('view_profile', account_id=session['user_id']))
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    return render_template_string(UPDATE_HTML, base=BASE_LAYOUT, user=user)

@app.route('/view_lookup', methods=['POST'])
def view_lookup():
    account_id = request.form.get('account_id')
    return redirect(url_for('view_profile', account_id=account_id))

@app.route('/profile/<int:account_id>')
def view_profile(account_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()
    return render_template_string(VIEW_HTML, base=BASE_LAYOUT, user=user)

@app.context_processor
def inject_base():
    return {'base': BASE_LAYOUT}

if __name__ == '__main__':
    app.run(debug=True)