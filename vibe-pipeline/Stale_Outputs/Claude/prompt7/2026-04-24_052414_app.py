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
        form { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input[type=text], input[type=email], input[type=password], textarea {
            width: 100%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;
        }
        input[type=submit] {
            margin-top: 15px; padding: 10px 20px; background: #4CAF50; color: white; border: none;
            border-radius: 4px; cursor: pointer; font-size: 16px;
        }
        input[type=submit]:hover { background: #45a049; }
        .nav { background: #333; padding: 10px; border-radius: 8px; margin-bottom: 20px; }
        .nav a { color: white; text-decoration: none; margin-right: 15px; font-size: 16px; }
        .nav a:hover { text-decoration: underline; }
        .alert { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .alert-success { background: #dff0d8; color: #3c763d; border: 1px solid #d6e9c6; }
        .alert-error { background: #f2dede; color: #a94442; border: 1px solid #ebccd1; }
        .profile-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .profile-card p { margin: 8px 0; }
        .profile-card strong { display: inline-block; width: 120px; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/view">View Profile</a>
    </div>
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="alert alert-{{ category }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
'''

HOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h1>Welcome to the User Profile App</h1>
<p>Use the navigation above to:</p>
<ul>
    <li><a href="/register">Create a new account</a></li>
    <li><a href="/update">Update your profile</a> (you'll need your account ID)</li>
    <li><a href="/view">View a profile by account ID</a></li>
</ul>
{% endblock %}
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
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
    <label>Phone Number:</label>
    <input type="text" name="phone">
    <label>Address:</label>
    <textarea name="address" rows="3"></textarea>
    <input type="submit" value="Create Account">
</form>
{% endblock %}
''')

UPDATE_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>Update Profile</h2>
<form method="POST" action="/update">
    <label>Account ID:</label>
    <input type="text" name="account_id" value="{{ user.id if user else '' }}" required>
    <label>Password (required to update):</label>
    <input type="password" name="password" required>
    <label>Full Name:</label>
    <input type="text" name="name" value="{{ user.name if user else '' }}">
    <label>Email:</label>
    <input type="email" name="email" value="{{ user.email if user else '' }}">
    <label>Phone Number:</label>
    <input type="text" name="phone" value="{{ user.phone if user else '' }}">
    <label>Address:</label>
    <textarea name="address" rows="3">{{ user.address if user else '' }}</textarea>
    <input type="submit" value="Update Profile">
</form>
{% endblock %}
''')

VIEW_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>View Profile by Account ID</h2>
<form method="GET" action="/view">
    <label>Account ID:</label>
    <input type="text" name="account_id" value="{{ account_id if account_id else '' }}">
    <input type="submit" value="View Profile">
</form>
{% if user %}
<br>
<div class="profile-card">
    <h3>Profile Details</h3>
    <p><strong>Account ID:</strong> {{ user.id }}</p>
    <p><strong>Username:</strong> {{ user.username }}</p>
    <p><strong>Name:</strong> {{ user.name or 'N/A' }}</p>
    <p><strong>Email:</strong> {{ user.email or 'N/A' }}</p>
    <p><strong>Phone:</strong> {{ user.phone or 'N/A' }}</p>
    <p><strong>Address:</strong> {{ user.address or 'N/A' }}</p>
</div>
{% endif %}
{% endblock %}
''')

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

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
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            flash(f'Account created successfully! Your Account ID is: {user["id"]}', 'success')
            return redirect(url_for('home'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.', 'error')
        finally:
            conn.close()

    return render_template_string(REGISTER_TEMPLATE)

@app.route('/update', methods=['GET', 'POST'])
def update():
    user = None
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not account_id or not password:
            flash('Account ID and password are required.', 'error')
            return render_template_string(UPDATE_TEMPLATE, user=None)

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()

        if not user:
            flash('Account not found.', 'error')
            conn.close()
            return render_template_string(UPDATE_TEMPLATE, user=None)

        if user['password'] != password:
            flash('Incorrect password.', 'error')
            conn.close()
            return render_template_string(UPDATE_TEMPLATE, user=None)

        conn.execute(
            'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
            (name, email, phone, address, account_id)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
        conn.close()
        flash('Profile updated successfully!', 'success')
        return render_template_string(UPDATE_TEMPLATE, user=user)

    account_id = request.args.get('account_id')
    if account_id:
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
        conn.close()

    return render_template_string(UPDATE_TEMPLATE, user=user)

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

    return render_template_string(VIEW_TEMPLATE, user=user, account_id=account_id)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)