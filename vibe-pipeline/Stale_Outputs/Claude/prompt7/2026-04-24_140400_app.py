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
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    label { display: block; margin-top: 10px; font-weight: bold; }
    input[type=text], input[type=email], input[type=password], textarea {
        width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box;
        border: 1px solid #ccc; border-radius: 4px;
    }
    button, input[type=submit] {
        margin-top: 15px; padding: 10px 20px; background: #4CAF50; color: white;
        border: none; border-radius: 4px; cursor: pointer; font-size: 16px;
    }
    button:hover, input[type=submit]:hover { background: #45a049; }
    .nav { margin-bottom: 20px; }
    .nav a { margin-right: 15px; color: #4CAF50; text-decoration: none; font-weight: bold; }
    .nav a:hover { text-decoration: underline; }
    .flash { background: #dff0d8; padding: 10px; border-radius: 4px; margin-bottom: 15px; color: #3c763d; }
    .flash.error { background: #f2dede; color: #a94442; }
    .profile-card { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .profile-card p { margin: 8px 0; }
    .profile-card strong { display: inline-block; width: 120px; }
</style>
"""

HOME_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/view">View Profile</a>
</div>
<h1>User Profile App</h1>
<p>Welcome! Use the navigation above to:</p>
<ul>
    <li><a href="/register">Create a new account</a></li>
    <li><a href="/view">View a profile by Account ID</a></li>
</ul>
"""

REGISTER_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/view">View Profile</a>
</div>
<h2>Create Account</h2>
{% for msg in messages %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
{% endfor %}
<form method="POST">
    <label>Name</label>
    <input type="text" name="name" placeholder="Full Name" required>
    <label>Email</label>
    <input type="email" name="email" placeholder="Email Address" required>
    <label>Phone</label>
    <input type="text" name="phone" placeholder="Phone Number">
    <label>Address</label>
    <textarea name="address" placeholder="Street Address" rows="3"></textarea>
    <label>Password</label>
    <input type="password" name="password" placeholder="Password" required>
    <input type="submit" value="Create Account">
</form>
"""

UPDATE_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/view">View Profile</a>
</div>
<h2>Update Profile (Account ID: {{ user['id'] }})</h2>
{% for msg in messages %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
{% endfor %}
<form method="POST">
    <label>Name</label>
    <input type="text" name="name" value="{{ user['name'] or '' }}" required>
    <label>Email</label>
    <input type="email" name="email" value="{{ user['email'] or '' }}" required>
    <label>Phone</label>
    <input type="text" name="phone" value="{{ user['phone'] or '' }}">
    <label>Address</label>
    <textarea name="address" rows="3">{{ user['address'] or '' }}</textarea>
    <label>New Password (leave blank to keep current)</label>
    <input type="password" name="password" placeholder="New Password">
    <input type="submit" value="Update Profile">
</form>
<br>
<a href="/profile/{{ user['id'] }}">View Profile</a>
"""

VIEW_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/view">View Profile</a>
</div>
<h2>View Profile by Account ID</h2>
{% for msg in messages %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
{% endfor %}
<form method="POST">
    <label>Account ID</label>
    <input type="text" name="account_id" placeholder="Enter Account ID" required>
    <input type="submit" value="View Profile">
</form>
"""

PROFILE_TEMPLATE = BASE_STYLE + """
<div class="nav">
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/view">View Profile</a>
</div>
<h2>Profile Details</h2>
<div class="profile-card">
    <p><strong>Account ID:</strong> {{ user['id'] }}</p>
    <p><strong>Name:</strong> {{ user['name'] or 'N/A' }}</p>
    <p><strong>Email:</strong> {{ user['email'] or 'N/A' }}</p>
    <p><strong>Phone:</strong> {{ user['phone'] or 'N/A' }}</p>
    <p><strong>Address:</strong> {{ user['address'] or 'N/A' }}</p>
</div>
<br>
<a href="/update/{{ user['id'] }}">Edit Profile</a>
"""

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    messages = []
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        password = request.form.get('password', '').strip()

        if not name or not email or not password:
            messages.append(('error', 'Name, email, and password are required.'))
        else:
            conn = get_db()
            cursor = conn.execute(
                'INSERT INTO users (name, email, phone, address, password) VALUES (?, ?, ?, ?, ?)',
                (name, email, phone, address, password)
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            messages.append(('', f'Account created successfully! Your Account ID is: {user_id}'))
    return render_template_string(REGISTER_TEMPLATE, messages=messages)

@app.route('/view', methods=['GET', 'POST'])
def view():
    messages = []
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        if account_id:
            return redirect(url_for('profile', user_id=account_id))
        else:
            messages.append(('error', 'Please enter an Account ID.'))
    return render_template_string(VIEW_TEMPLATE, messages=messages)

@app.route('/profile/<int:user_id>')
def profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if not user:
        return render_template_string(VIEW_TEMPLATE, messages=[('error', f'No user found with ID {user_id}.')])
    return render_template_string(PROFILE_TEMPLATE, user=user)

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update(user_id):
    messages = []
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        return render_template_string(VIEW_TEMPLATE, messages=[('error', f'No user found with ID {user_id}.')])

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        password = request.form.get('password', '').strip()

        if not name or not email:
            messages.append(('error', 'Name and email are required.'))
        else:
            conn = get_db()
            if password:
                conn.execute(
                    'UPDATE users SET name=?, email=?, phone=?, address=?, password=? WHERE id=?',
                    (name, email, phone, address, password, user_id)
                )
            else:
                conn.execute(
                    'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                    (name, email, phone, address, user_id)
                )
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
            conn.close()
            messages.append(('', 'Profile updated successfully!'))

    return render_template_string(UPDATE_TEMPLATE, user=user, messages=messages)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)