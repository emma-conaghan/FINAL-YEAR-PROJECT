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
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    input[type=text], input[type=password], input[type=email], textarea {
        width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px;
    }
    input[type=submit], .btn {
        background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block;
    }
    input[type=submit]:hover, .btn:hover { background: #45a049; }
    .btn-secondary { background: #2196F3; }
    .btn-secondary:hover { background: #1976D2; }
    .flash { padding: 10px; background: #ffeb3b; border-left: 4px solid #fbc02d; margin-bottom: 15px; }
    .error { background: #ffcdd2; border-left-color: #e53935; }
    .success { background: #c8e6c9; border-left-color: #388e3c; }
    .card { background: white; padding: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    label { font-weight: bold; color: #555; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 15px; color: #2196F3; text-decoration: none; }
    .profile-field { margin-bottom: 12px; }
    .profile-field span { font-weight: bold; }
</style>
"""

HOME_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h1>User Profile App</h1>
    <p>Welcome! You can create an account and manage your profile information.</p>
    <a href="/register" class="btn">Create Account</a>
    <a href="/view" class="btn btn-secondary" style="margin-left:10px;">View Profile by ID</a>
</div>
"""

REGISTER_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>Create Account</h2>
    {% for msg in messages %}
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

UPDATE_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>Update Profile - Account #{{ user['id'] }}</h2>
    {% for msg in messages %}
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
    <br>
    <a href="/profile/{{ user['id'] }}" class="btn btn-secondary">View Profile</a>
</div>
"""

VIEW_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>View Profile by Account ID</h2>
    {% for msg in messages %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Account ID:</label>
        <input type="text" name="account_id" required>
        <input type="submit" value="View Profile">
    </form>
</div>
"""

PROFILE_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>Profile Details</h2>
    <div class="profile-field"><span>Account ID:</span> {{ user['id'] }}</div>
    <div class="profile-field"><span>Username:</span> {{ user['username'] }}</div>
    <div class="profile-field"><span>Name:</span> {{ user['name'] or 'Not provided' }}</div>
    <div class="profile-field"><span>Email:</span> {{ user['email'] or 'Not provided' }}</div>
    <div class="profile-field"><span>Phone:</span> {{ user['phone'] or 'Not provided' }}</div>
    <div class="profile-field"><span>Address:</span> {{ user['address'] or 'Not provided' }}</div>
    <br>
    <a href="/update/{{ user['id'] }}" class="btn">Update Profile</a>
</div>
"""

UPDATE_AUTH_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>Authenticate to Update Profile</h2>
    {% for msg in messages %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <p>Please enter your credentials to update Account #{{ account_id }}</p>
    <form method="POST">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Authenticate">
    </form>
</div>
"""

def get_messages():
    msgs = []
    if '_flashes' in request.__dict__:
        pass
    return msgs

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    messages = []
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not username or not password:
            messages.append(('error', 'Username and password are required.'))
        else:
            try:
                conn = get_db()
                conn.execute(
                    'INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                    (username, password, name, email, phone, address)
                )
                conn.commit()
                user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
                conn.close()
                messages.append(('success', f'Account created! Your Account ID is {user["id"]}.'))
            except sqlite3.IntegrityError:
                messages.append(('error', 'Username already exists. Please choose another.'))
                conn.close()

    return render_template_string(REGISTER_TEMPLATE, messages=messages)

@app.route('/view', methods=['GET', 'POST'])
def view():
    messages = []
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        if not account_id:
            messages.append(('error', 'Please enter an Account ID.'))
        else:
            try:
                aid = int(account_id)
                return redirect(url_for('profile', account_id=aid))
            except ValueError:
                messages.append(('error', 'Account ID must be a number.'))
    return render_template_string(VIEW_TEMPLATE, messages=messages)

@app.route('/profile/<int:account_id>')
def profile(account_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()
    if not user:
        return render_template_string(BASE_STYLE + '<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav><div class="card"><h2>Not Found</h2><p>No account found with that ID.</p><a href="/view" class="btn btn-secondary">Try Again</a></div>')
    return render_template_string(PROFILE_TEMPLATE, user=user)

@app.route('/update/<int:account_id>', methods=['GET', 'POST'])
def update_auth(account_id):
    messages = []
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()
    if not user:
        return render_template_string(BASE_STYLE + '<div class="card"><h2>Not Found</h2><p>No account found with that ID.</p></div>')

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if user['username'] == username and user['password'] == password:
            return redirect(url_for('update_profile', account_id=account_id))
        else:
            messages.append(('error', 'Invalid username or password.'))

    return render_template_string(UPDATE_AUTH_TEMPLATE, account_id=account_id, messages=messages)

@app.route('/update/<int:account_id>/edit', methods=['GET', 'POST'])
def update_profile(account_id):
    messages = []
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    if not user:
        conn.close()
        return render_template_string(BASE_STYLE + '<div class="card"><h2>Not Found</h2><p>No account found with that ID.</p></div>')

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        conn.execute(
            'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
            (name, email, phone, address, account_id)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
        messages.append(('success', 'Profile updated successfully!'))

    conn.close()
    return render_template_string(UPDATE_TEMPLATE, user=user, messages=messages)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)