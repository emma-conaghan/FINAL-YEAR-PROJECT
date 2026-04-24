import sqlite3
from flask import Flask, request, render_template_string, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'simple_app_key'

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  name TEXT,
                  email TEXT,
                  phone TEXT,
                  address TEXT)''')
    conn.commit()
    conn.close()

init_db()

LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Profile App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .container { max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 20px; border-radius: 8px; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input, textarea { width: 100%; padding: 8px; margin-top: 5px; box-sizing: border-box; }
        button { margin-top: 20px; padding: 10px 15px; background: #28a745; color: white; border: none; cursor: pointer; }
        nav { margin-bottom: 20px; }
        .msg { color: blue; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> |
        {% if session.get('user_id') %}
            <a href="{{ url_for('profile') }}">Edit Profile</a> |
            <a href="{{ url_for('logout') }}">Logout</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> |
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    <div class="container">
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            {% for message in messages %}
              <p class="msg">{{ message }}</p>
            {% endfor %}
          {% endif %}
        {% endwith %}
        {% block content %}{% endblock content %}
    </div>
</body>
</html>
"""

INDEX_HTML = """
{% extends "layout" %}
{% block content %}
<h1>Welcome to User Profiles</h1>
<p>Search for a profile by ID:</p>
<form action="{{ url_for('view_user') }}" method="get">
    <input type="number" name="account_id" placeholder="Account ID" required>
    <button type="submit">View Profile</button>
</form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Create Account</h2>
<form method="post">
    <label>Username</label><input name="username" required>
    <label>Password</label><input type="password" name="password" required>
    <button type="submit">Register</button>
</form>
{% endblock %}
"""

LOGIN_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Login</h2>
<form method="post">
    <label>Username</label><input name="username" required>
    <label>Password</label><input type="password" name="password" required>
    <button type="submit">Login</button>
</form>
{% endblock %}
"""

PROFILE_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Update Profile Information</h2>
<form method="post">
    <label>Full Name</label><input name="name" value="{{ user['name'] or '' }}">
    <label>Email</label><input type="email" name="email" value="{{ user['email'] or '' }}">
    <label>Phone Number</label><input name="phone" value="{{ user['phone'] or '' }}">
    <label>Address</label><textarea name="address">{{ user['address'] or '' }}</textarea>
    <button type="submit">Update Info</button>
</form>
<p>Your Account ID is: <strong>{{ user['id'] }}</strong></p>
{% endblock %}
"""

VIEW_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Profile Details</h2>
{% if user %}
    <p><strong>Name:</strong> {{ user['name'] or 'N/A' }}</p>
    <p><strong>Email:</strong> {{ user['email'] or 'N/A' }}</p>
    <p><strong>Phone:</strong> {{ user['phone'] or 'N/A' }}</p>
    <p><strong>Address:</strong> {{ user['address'] or 'N/A' }}</p>
{% else %}
    <p>User not found.</p>
{% endif %}
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(LAYOUT + INDEX_HTML)

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
            return "Username already exists"
        finally:
            conn.close()
    return render_template_string(LAYOUT + REGISTER_HTML)

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
        return "Invalid credentials"
    return render_template_string(LAYOUT + LOGIN_HTML)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    if request.method == 'POST':
        conn.execute('UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
                     (request.form['name'], request.form['email'], request.form['phone'], request.form['address'], session['user_id']))
        conn.commit()
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    return render_template_string(LAYOUT + PROFILE_HTML, user=user)

@app.route('/view_user')
def view_user():
    account_id = request.args.get('account_id')
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()
    return render_template_string(LAYOUT + VIEW_HTML, user=user)

@app.route('/user/<int:account_id>')
def view_user_direct(account_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()
    return render_template_string(LAYOUT + VIEW_HTML, user=user)

if __name__ == '__main__':
    app.run(debug=True)