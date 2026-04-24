from flask import Flask, render_template_string, request, redirect, url_for, flash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'
DB_PATH = 'users.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    input[type=text], input[type=password], input[type=email], textarea {
        width: 100%; padding: 8px; margin: 5px 0 15px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px;
    }
    input[type=submit] {
        background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;
    }
    input[type=submit]:hover { background: #45a049; }
    .flash { padding: 10px; margin: 10px 0; border-radius: 4px; }
    .flash.error { background: #f8d7da; color: #721c24; }
    .flash.success { background: #d4edda; color: #155724; }
    a { color: #4CAF50; }
    .nav { margin-bottom: 20px; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    label { font-weight: bold; }
    .profile-field { margin-bottom: 10px; }
    .profile-label { font-weight: bold; color: #555; }
</style>
"""

INDEX_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/register">Register</a> | <a href="/login">Login</a> | <a href="/view">View Profile by ID</a>
</div>
<div class="card">
    <h1>Welcome to User Profile App</h1>
    <p>Create an account, update your profile, or view profiles by ID.</p>
</div>
"""

REGISTER_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a> | <a href="/login">Login</a>
</div>
<div class="card">
    <h2>Create Account</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <label>Name:</label>
        <input type="text" name="name">
        <label>Email:</label>
        <input type="text" name="email">
        <label>Phone:</label>
        <input type="text" name="phone">
        <label>Address:</label>
        <input type="text" name="address">
        <input type="submit" value="Register">
    </form>
</div>
"""

LOGIN_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a> | <a href="/register">Register</a>
</div>
<div class="card">
    <h2>Login</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Login">
    </form>
</div>
"""

PROFILE_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a> | <a href="/view">View Profile by ID</a> | <a href="/logout">Logout</a>
</div>
<div class="card">
    <h2>Edit Profile (Account ID: {{ user['id'] }})</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" value="{{ user['name'] or '' }}">
        <label>Email:</label>
        <input type="text" name="email" value="{{ user['email'] or '' }}">
        <label>Phone:</label>
        <input type="text" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address:</label>
        <input type="text" name="address" value="{{ user['address'] or '' }}">
        <input type="submit" value="Update Profile">
    </form>
</div>
"""

VIEW_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a> | <a href="/register">Register</a> | <a href="/login">Login</a>
</div>
<div class="card">
    <h2>View Profile by Account ID</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="GET">
        <label>Account ID:</label>
        <input type="text" name="account_id">
        <input type="submit" value="View Profile">
    </form>
    {% if user %}
    <hr>
    <h3>Profile Details</h3>
    <div class="profile-field"><span class="profile-label">Account ID:</span> {{ user['id'] }}</div>
    <div class="profile-field"><span class="profile-label">Username:</span> {{ user['username'] }}</div>
    <div class="profile-field"><span class="profile-label">Name:</span> {{ user['name'] or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Email:</span> {{ user['email'] or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Phone:</span> {{ user['phone'] or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Address:</span> {{ user['address'] or 'Not set' }}</div>
    {% endif %}
</div>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template_string(REGISTER_TEMPLATE)

        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                (username, password, name, email, phone, address)
            )
            conn.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'error')
        finally:
            conn.close()

    return render_template_string(REGISTER_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ? AND password = ?',
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            from flask import session
            session['user_id'] = user['id']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('profile'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template_string(LOGIN_TEMPLATE)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    from flask import session
    user_id = session.get('user_id')
    if not user_id:
        flash('Please log in to view your profile.', 'error')
        return redirect(url_for('login'))

    conn = get_db()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        conn.execute(
            'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        flash('Profile updated successfully!', 'success')

    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return render_template_string(PROFILE_TEMPLATE, user=user)

@app.route('/view', methods=['GET'])
def view():
    account_id = request.args.get('account_id', '').strip()
    user = None
    if account_id:
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
        conn.close()
        if not user:
            flash('No account found with that ID.', 'error')
    return render_template_string(VIEW_TEMPLATE, user=user)

@app.route('/logout')
def logout():
    from flask import session
    session.clear()
    flash('Logged out.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)