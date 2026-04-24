from flask import Flask, render_template_string, request, redirect, url_for, flash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'
DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
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

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
    h1, h2 { color: #333; }
    form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    input[type=text], input[type=password], input[type=email], textarea {
        width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box;
        border: 1px solid #ccc; border-radius: 4px;
    }
    input[type=submit] {
        background: #4CAF50; color: white; padding: 10px 20px;
        border: none; border-radius: 4px; cursor: pointer;
    }
    input[type=submit]:hover { background: #45a049; }
    a { color: #4CAF50; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .flash { background: #ffe0e0; color: #a00; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
    .success { background: #e0ffe0; color: #080; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
    .profile-box { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .profile-box p { margin: 8px 0; }
    .label { font-weight: bold; color: #555; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 15px; }
</style>
'''

INDEX_TEMPLATE = BASE_STYLE + '''
<h1>User Profile App</h1>
<nav>
    <a href="/register">Register</a>
    <a href="/view">View Profile by ID</a>
</nav>
<p>Welcome! Use the links above to register an account or view a profile.</p>
'''

REGISTER_TEMPLATE = BASE_STYLE + '''
<h1>Create Account</h1>
<nav><a href="/">Home</a> | <a href="/view">View Profile</a></nav>
{% for msg in get_flashed_messages() %}
  <div class="flash">{{ msg }}</div>
{% endfor %}
{% if success %}
  <div class="success">{{ success }}</div>
{% endif %}
<form method="POST">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Password:</label>
    <input type="password" name="password" required>
    <label>Full Name:</label>
    <input type="text" name="name">
    <label>Email:</label>
    <input type="text" name="email">
    <label>Phone Number:</label>
    <input type="text" name="phone">
    <label>Address:</label>
    <input type="text" name="address">
    <input type="submit" value="Register">
</form>
'''

UPDATE_TEMPLATE = BASE_STYLE + '''
<h1>Update Profile</h1>
<nav><a href="/">Home</a> | <a href="/view">View Profile</a></nav>
{% for msg in get_flashed_messages() %}
  <div class="flash">{{ msg }}</div>
{% endfor %}
{% if success %}
  <div class="success">{{ success }}</div>
{% endif %}
<form method="POST">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Password (to verify):</label>
    <input type="password" name="password" required>
    <label>New Full Name:</label>
    <input type="text" name="name" value="{{ user.name if user else '' }}">
    <label>New Email:</label>
    <input type="text" name="email" value="{{ user.email if user else '' }}">
    <label>New Phone Number:</label>
    <input type="text" name="phone" value="{{ user.phone if user else '' }}">
    <label>New Address:</label>
    <input type="text" name="address" value="{{ user.address if user else '' }}">
    <input type="submit" value="Update Profile">
</form>
'''

VIEW_TEMPLATE = BASE_STYLE + '''
<h1>View Profile by Account ID</h1>
<nav><a href="/">Home</a> | <a href="/register">Register</a> | <a href="/update">Update Profile</a></nav>
{% for msg in get_flashed_messages() %}
  <div class="flash">{{ msg }}</div>
{% endfor %}
<form method="GET">
    <label>Enter Account ID:</label>
    <input type="text" name="id" value="{{ account_id }}">
    <input type="submit" value="View Profile">
</form>
{% if user %}
<div class="profile-box" style="margin-top:20px;">
    <h2>Profile Details</h2>
    <p><span class="label">Account ID:</span> {{ user.id }}</p>
    <p><span class="label">Username:</span> {{ user.username }}</p>
    <p><span class="label">Full Name:</span> {{ user.name or 'Not provided' }}</p>
    <p><span class="label">Email:</span> {{ user.email or 'Not provided' }}</p>
    <p><span class="label">Phone:</span> {{ user.phone or 'Not provided' }}</p>
    <p><span class="label">Address:</span> {{ user.address or 'Not provided' }}</p>
</div>
{% endif %}
'''

@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not username or not password:
            flash('Username and password are required.')
            return render_template_string(REGISTER_TEMPLATE, success=None)

        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                (username, password, name, email, phone, address)
            )
            conn.commit()
            cur = conn.execute('SELECT id FROM users WHERE username = ?', (username,))
            row = cur.fetchone()
            success = f'Account created successfully! Your Account ID is: {row["id"]}. Use this ID to view your profile.'
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose a different username.')
        finally:
            conn.close()

    return render_template_string(REGISTER_TEMPLATE, success=success)

@app.route('/update', methods=['GET', 'POST'])
def update():
    success = None
    user = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not username or not password:
            flash('Username and password are required.')
            return render_template_string(UPDATE_TEMPLATE, success=None, user=None)

        conn = get_db()
        cur = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cur.fetchone()

        if not user:
            flash('Invalid username or password.')
            conn.close()
            return render_template_string(UPDATE_TEMPLATE, success=None, user=None)

        conn.execute(
            'UPDATE users SET name=?, email=?, phone=?, address=? WHERE username=? AND password=?',
            (name, email, phone, address, username, password)
        )
        conn.commit()
        conn.close()
        success = 'Profile updated successfully!'

    return render_template_string(UPDATE_TEMPLATE, success=success, user=user)

@app.route('/view', methods=['GET'])
def view():
    account_id = request.args.get('id', '').strip()
    user = None
    if account_id:
        conn = get_db()
        cur = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,))
        user = cur.fetchone()
        conn.close()
        if not user:
            flash(f'No account found with ID: {account_id}')
    return render_template_string(VIEW_TEMPLATE, user=user, account_id=account_id)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)