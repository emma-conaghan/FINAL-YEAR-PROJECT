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

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>User Profile App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1, h2 { color: #333; }
        form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        input[type=text], input[type=password], input[type=email], textarea {
            width: 100%; padding: 8px; margin: 6px 0 12px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px;
        }
        input[type=submit] {
            background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;
        }
        input[type=submit]:hover { background: #45a049; }
        .nav { margin-bottom: 20px; }
        .nav a { margin-right: 15px; color: #4CAF50; text-decoration: none; font-weight: bold; }
        .nav a:hover { text-decoration: underline; }
        .flash { background: #dff0d8; color: #3c763d; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .error { background: #f2dede; color: #a94442; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .profile-card { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .profile-card p { margin: 8px 0; }
        .profile-card strong { display: inline-block; width: 120px; }
        label { font-weight: bold; color: #555; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/update">Update Profile</a>
        <a href="/view">View Profile</a>
    </div>
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% for category, message in messages %}
            <div class="{{ 'error' if category == 'error' else 'flash' }}">{{ message }}</div>
        {% endfor %}
    {% endwith %}
    {{ content }}
</body>
</html>
'''

HOME_CONTENT = '''
<h1>Welcome to User Profile App</h1>
<p>Use the navigation above to:</p>
<ul>
    <li><a href="/register">Create a new account</a></li>
    <li><a href="/update">Update your profile</a></li>
    <li><a href="/view">View a profile by Account ID</a></li>
</ul>
'''

REGISTER_CONTENT = '''
<h2>Create Account</h2>
<form method="POST" action="/register">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Password:</label>
    <input type="password" name="password" required>
    <label>Full Name:</label>
    <input type="text" name="name">
    <label>Email:</label>
    <input type="email" name="email">
    <label>Phone:</label>
    <input type="text" name="phone">
    <label>Address:</label>
    <input type="text" name="address">
    <input type="submit" value="Register">
</form>
'''

UPDATE_CONTENT = '''
<h2>Update Profile</h2>
<form method="POST" action="/update">
    <label>Account ID:</label>
    <input type="text" name="account_id" required>
    <label>Password (to verify):</label>
    <input type="password" name="password" required>
    <label>Full Name:</label>
    <input type="text" name="name">
    <label>Email:</label>
    <input type="email" name="email">
    <label>Phone:</label>
    <input type="text" name="phone">
    <label>Address:</label>
    <input type="text" name="address">
    <input type="submit" value="Update">
</form>
'''

VIEW_CONTENT = '''
<h2>View Profile by Account ID</h2>
<form method="GET" action="/view">
    <label>Account ID:</label>
    <input type="text" name="account_id" required>
    <input type="submit" value="View">
</form>
{% if user %}
<br>
<div class="profile-card">
    <h3>Profile Details</h3>
    <p><strong>Account ID:</strong> {{ user['id'] }}</p>
    <p><strong>Username:</strong> {{ user['username'] }}</p>
    <p><strong>Name:</strong> {{ user['name'] or 'N/A' }}</p>
    <p><strong>Email:</strong> {{ user['email'] or 'N/A' }}</p>
    <p><strong>Phone:</strong> {{ user['phone'] or 'N/A' }}</p>
    <p><strong>Address:</strong> {{ user['address'] or 'N/A' }}</p>
</div>
{% endif %}
'''

def render_page(content_html, **kwargs):
    from jinja2 import Template
    content_template = Template(content_html)
    rendered_content = content_template.render(**kwargs)
    full_template = BASE_TEMPLATE.replace('{{ content }}', rendered_content)
    return render_template_string(full_template)

@app.route('/')
def home():
    return render_page(HOME_CONTENT)

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
            return render_page(REGISTER_CONTENT)

        try:
            conn = get_db()
            conn.execute(
                'INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                (username, password, name, email, phone, address)
            )
            conn.commit()
            user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            conn.close()
            flash(f'Account created successfully! Your Account ID is: {user["id"]}', 'success')
            return redirect(url_for('home'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose a different one.', 'error')
            return render_page(REGISTER_CONTENT)

    return render_page(REGISTER_CONTENT)

@app.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not account_id or not password:
            flash('Account ID and password are required.', 'error')
            return render_page(UPDATE_CONTENT)

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()

        if not user:
            conn.close()
            flash('Account not found.', 'error')
            return render_page(UPDATE_CONTENT)

        if user['password'] != password:
            conn.close()
            flash('Incorrect password.', 'error')
            return render_page(UPDATE_CONTENT)

        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, account_id)
        )
        conn.commit()
        conn.close()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('home'))

    return render_page(UPDATE_CONTENT)

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

    return render_page(VIEW_CONTENT, user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)