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
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT,
                address TEXT,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>User Profiles</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1, h2 { color: #333; }
        .nav { margin-bottom: 20px; }
        .nav a { margin-right: 15px; color: #0066cc; text-decoration: none; }
        .nav a:hover { text-decoration: underline; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button { background: #0066cc; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0055aa; }
        .flash { padding: 10px; border-radius: 4px; margin-bottom: 15px; }
        .flash.success { background: #d4edda; color: #155724; }
        .flash.error { background: #f8d7da; color: #721c24; }
        label { font-weight: bold; color: #555; }
        .profile-field { margin-bottom: 10px; }
        .profile-field span { color: #333; }
        .profile-label { font-weight: bold; color: #555; display: inline-block; width: 120px; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/view">View Profile by ID</a>
    </div>
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="flash {{ category }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
'''

HOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<div class="card">
    <h1>User Profile App</h1>
    <p>Welcome! You can:</p>
    <ul>
        <li><a href="/register">Create a new account</a></li>
        <li><a href="/view">View a profile by Account ID</a></li>
        <li>Update your profile by going to <a href="/update">/update</a></li>
    </ul>
</div>
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<div class="card">
    <h2>Create Account</h2>
    <form method="POST" action="/register">
        <label>Name:</label>
        <input type="text" name="name" required>
        <label>Email:</label>
        <input type="email" name="email" required>
        <label>Phone:</label>
        <input type="text" name="phone">
        <label>Address:</label>
        <textarea name="address" rows="3"></textarea>
        <label>Password:</label>
        <input type="password" name="password" required>
        <button type="submit">Create Account</button>
    </form>
</div>
''')

UPDATE_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<div class="card">
    <h2>Update Profile</h2>
    <p>Enter your Account ID and email to find and update your profile.</p>
    <form method="POST" action="/update">
        <label>Account ID:</label>
        <input type="number" name="account_id" required>
        <label>Email (for verification):</label>
        <input type="email" name="email_verify" required>
        <label>New Name:</label>
        <input type="text" name="name">
        <label>New Email:</label>
        <input type="email" name="email">
        <label>New Phone:</label>
        <input type="text" name="phone">
        <label>New Address:</label>
        <textarea name="address" rows="3"></textarea>
        <label>New Password (leave blank to keep current):</label>
        <input type="password" name="password">
        <button type="submit">Update Profile</button>
    </form>
</div>
''')

VIEW_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<div class="card">
    <h2>View Profile by Account ID</h2>
    <form method="GET" action="/view">
        <label>Account ID:</label>
        <input type="number" name="account_id" value="{{ account_id or '' }}" required>
        <button type="submit">View Profile</button>
    </form>
</div>
{% if user %}
<div class="card">
    <h2>Profile Details</h2>
    <div class="profile-field"><span class="profile-label">Account ID:</span> <span>{{ user['id'] }}</span></div>
    <div class="profile-field"><span class="profile-label">Name:</span> <span>{{ user['name'] }}</span></div>
    <div class="profile-field"><span class="profile-label">Email:</span> <span>{{ user['email'] }}</span></div>
    <div class="profile-field"><span class="profile-label">Phone:</span> <span>{{ user['phone'] or 'N/A' }}</span></div>
    <div class="profile-field"><span class="profile-label">Address:</span> <span>{{ user['address'] or 'N/A' }}</span></div>
    <br>
    <a href="/update">Update this profile</a>
</div>
{% elif account_id %}
<div class="card">
    <p>No user found with Account ID: {{ account_id }}</p>
</div>
{% endif %}
''')

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        password = request.form.get('password', '')

        if not name or not email or not password:
            flash('Name, email, and password are required.', 'error')
            return render_template_string(REGISTER_TEMPLATE)

        try:
            with get_db() as conn:
                cursor = conn.execute(
                    'INSERT INTO users (name, email, phone, address, password) VALUES (?, ?, ?, ?, ?)',
                    (name, email, phone, address, password)
                )
                conn.commit()
                user_id = cursor.lastrowid
                flash(f'Account created successfully! Your Account ID is: {user_id}', 'success')
                return redirect(url_for('home'))
        except sqlite3.IntegrityError:
            flash('Email already exists. Please use a different email.', 'error')
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        email_verify = request.form.get('email_verify', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        password = request.form.get('password', '')

        if not account_id or not email_verify:
            flash('Account ID and email are required for verification.', 'error')
            return render_template_string(UPDATE_TEMPLATE)

        with get_db() as conn:
            user = conn.execute('SELECT * FROM users WHERE id = ? AND email = ?', (account_id, email_verify)).fetchone()
            if not user:
                flash('No account found with that ID and email combination.', 'error')
                return render_template_string(UPDATE_TEMPLATE)

            new_name = name if name else user['name']
            new_email = email if email else user['email']
            new_phone = phone if phone else user['phone']
            new_address = address if address else user['address']
            new_password = password if password else user['password']

            try:
                conn.execute(
                    'UPDATE users SET name=?, email=?, phone=?, address=?, password=? WHERE id=?',
                    (new_name, new_email, new_phone, new_address, new_password, account_id)
                )
                conn.commit()
                flash('Profile updated successfully!', 'success')
                return redirect(url_for('view_profile') + f'?account_id={account_id}')
            except sqlite3.IntegrityError:
                flash('Email already taken by another account.', 'error')

    return render_template_string(UPDATE_TEMPLATE)

@app.route('/view', methods=['GET'])
def view_profile():
    account_id = request.args.get('account_id', '').strip()
    user = None
    if account_id:
        with get_db() as conn:
            user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    return render_template_string(VIEW_TEMPLATE, user=user, account_id=account_id)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)