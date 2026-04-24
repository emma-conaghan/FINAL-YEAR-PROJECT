from flask import Flask, render_template_string, request, redirect, url_for
import sqlite3
import os

app = Flask(__name__)
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
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT DEFAULT '',
            address TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f6f9; color: #333; }
        nav { background: #2c3e50; padding: 15px 30px; display: flex; align-items: center; gap: 20px; }
        nav a { color: #ecf0f1; text-decoration: none; font-size: 16px; }
        nav a:hover { text-decoration: underline; }
        nav .brand { font-size: 20px; font-weight: bold; margin-right: 20px; }
        .container { max-width: 700px; margin: 40px auto; padding: 0 20px; }
        .card { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 30px; margin-bottom: 20px; }
        h1 { margin-bottom: 20px; color: #2c3e50; }
        h2 { margin-bottom: 15px; color: #2c3e50; }
        label { display: block; margin-bottom: 5px; font-weight: bold; margin-top: 15px; }
        input, textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        textarea { resize: vertical; min-height: 80px; }
        button, .btn { background: #2c3e50; color: #fff; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 20px; display: inline-block; text-decoration: none; }
        button:hover, .btn:hover { background: #34495e; }
        .btn-secondary { background: #7f8c8d; }
        .btn-secondary:hover { background: #95a5a6; }
        .profile-field { padding: 10px 0; border-bottom: 1px solid #eee; }
        .profile-field:last-child { border-bottom: none; }
        .profile-label { font-weight: bold; color: #7f8c8d; font-size: 13px; text-transform: uppercase; }
        .profile-value { margin-top: 4px; font-size: 16px; }
        .message { padding: 12px; border-radius: 4px; margin-bottom: 20px; }
        .message.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .search-form { display: flex; gap: 10px; align-items: end; }
        .search-form input { flex: 1; }
        .search-form button { margin-top: 0; }
        .user-list { list-style: none; }
        .user-list li { padding: 12px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .user-list li:last-child { border-bottom: none; }
        .user-list a { color: #2c3e50; text-decoration: none; }
        .user-list a:hover { text-decoration: underline; }
        .id-badge { background: #2c3e50; color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 13px; }
    </style>
</head>
<body>
    <nav>
        <span class="brand">👤 ProfileApp</span>
        <a href="/">Home</a>
        <a href="/create">Create Account</a>
        <a href="/lookup">View Profile</a>
        <a href="/users">All Users</a>
    </nav>
    <div class="container">
        {{ content }}
    </div>
</body>
</html>
'''

HOME_CONTENT = '''
<h1>Welcome to ProfileApp</h1>
<div class="card">
    <h2>Manage Your Profile</h2>
    <p style="margin-bottom:20px;">Create an account, update your information, or look up a profile by account ID.</p>
    <a href="/create" class="btn">Create Account</a>
    <a href="/lookup" class="btn btn-secondary" style="margin-left:10px;">View Profile</a>
</div>
'''

CREATE_CONTENT = '''
<h1>Create Account</h1>
<div class="card">
    {% if message %}
    <div class="message {{ message_type }}">{{ message }}</div>
    {% endif %}
    <form method="POST" action="/create">
        <label for="name">Full Name *</label>
        <input type="text" id="name" name="name" required placeholder="Enter your full name">

        <label for="email">Email Address *</label>
        <input type="email" id="email" name="email" required placeholder="Enter your email">

        <label for="phone">Phone Number</label>
        <input type="tel" id="phone" name="phone" placeholder="Enter your phone number">

        <label for="address">Address</label>
        <textarea id="address" name="address" placeholder="Enter your address"></textarea>

        <button type="submit">Create Account</button>
    </form>
</div>
'''

LOOKUP_CONTENT = '''
<h1>View Profile</h1>
<div class="card">
    <form method="GET" action="/profile">
        <label for="account_id">Account ID</label>
        <div class="search-form">
            <input type="number" id="account_id" name="id" required placeholder="Enter account ID" min="1">
            <button type="submit">Look Up</button>
        </div>
    </form>
</div>
'''

PROFILE_CONTENT = '''
<h1>Profile Details</h1>
{% if message %}
<div class="message {{ message_type }}">{{ message }}</div>
{% endif %}
{% if user %}
<div class="card">
    <div class="profile-field">
        <div class="profile-label">Account ID</div>
        <div class="profile-value"><span class="id-badge">#{{ user.id }}</span></div>
    </div>
    <div class="profile-field">
        <div class="profile-label">Full Name</div>
        <div class="profile-value">{{ user.name }}</div>
    </div>
    <div class="profile-field">
        <div class="profile-label">Email Address</div>
        <div class="profile-value">{{ user.email }}</div>
    </div>
    <div class="profile-field">
        <div class="profile-label">Phone Number</div>
        <div class="profile-value">{{ user.phone if user.phone else 'Not provided' }}</div>
    </div>
    <div class="profile-field">
        <div class="profile-label">Address</div>
        <div class="profile-value">{{ user.address if user.address else 'Not provided' }}</div>
    </div>
    <a href="/edit/{{ user.id }}" class="btn">Edit Profile</a>
</div>
{% else %}
<div class="card">
    <p>No user found with that account ID.</p>
    <a href="/lookup" class="btn btn-secondary">Try Again</a>
</div>
{% endif %}
'''

EDIT_CONTENT = '''
<h1>Edit Profile</h1>
<div class="card">
    {% if message %}
    <div class="message {{ message_type }}">{{ message }}</div>
    {% endif %}
    {% if user %}
    <p style="margin-bottom:10px;">Account ID: <span class="id-badge">#{{ user.id }}</span></p>
    <form method="POST" action="/edit/{{ user.id }}">
        <label for="name">Full Name *</label>
        <input type="text" id="name" name="name" required value="{{ user.name }}">

        <label for="email">Email Address *</label>
        <input type="email" id="email" name="email" required value="{{ user.email }}">

        <label for="phone">Phone Number</label>
        <input type="tel" id="phone" name="phone" value="{{ user.phone }}">

        <label for="address">Address</label>
        <textarea id="address" name="address">{{ user.address }}</textarea>

        <button type="submit">Save Changes</button>
        <a href="/profile?id={{ user.id }}" class="btn btn-secondary" style="margin-left:10px;">Cancel</a>
    </form>
    {% else %}
    <p>User not found.</p>
    <a href="/lookup" class="btn btn-secondary">Go Back</a>
    {% endif %}
</div>
'''

USERS_CONTENT = '''
<h1>All Users</h1>
<div class="card">
    {% if users %}
    <ul class="user-list">
        {% for user in users %}
        <li>
            <span><span class="id-badge">#{{ user.id }}</span> &nbsp; <a href="/profile?id={{ user.id }}">{{ user.name }}</a> &mdash; {{ user.email }}</span>
            <a href="/edit/{{ user.id }}" class="btn" style="margin-top:0; padding:6px 14px; font-size:13px;">Edit</a>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <p>No users yet. <a href="/create">Create one!</a></p>
    {% endif %}
</div>
'''


def render_page(title, content_template, **kwargs):
    content = render_template_string(content_template, **kwargs)
    return render_template_string(BASE_TEMPLATE, title=title, content=content)


@app.route('/')
def home():
    return render_page('ProfileApp - Home', HOME_CONTENT)


@app.route('/create', methods=['GET', 'POST'])
def create():
    message = None
    message_type = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            message = 'Name and email are required.'
            message_type = 'error'
        else:
            conn = get_db()
            cursor = conn.execute(
                'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
                (name, email, phone, address)
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            message = f'Account created successfully! Your Account ID is #{user_id}'
            message_type = 'success'

    return render_page('Create Account', CREATE_CONTENT, message=message, message_type=message_type)


@app.route('/lookup')
def lookup():
    return render_page('View Profile', LOOKUP_CONTENT)


@app.route('/profile')
def profile():
    user_id = request.args.get('id')
    user = None
    message = None
    message_type = None

    if user_id:
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        if not user:
            message = f'No user found with Account ID #{user_id}.'
            message_type = 'error'
    else:
        return redirect(url_for('lookup'))

    return render_page('Profile Details', PROFILE_CONTENT, user=user, message=message, message_type=message_type)


@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit(user_id):
    conn = get_db()
    message = None
    message_type = None

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            message = 'Name and email are required.'
            message_type = 'error'
        else:
            conn.execute(
                'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
                (name, email, phone, address, user_id)
            )
            conn.commit()
            message = 'Profile updated successfully!'
            message_type = 'success'

    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    return render_page('Edit Profile', EDIT_CONTENT, user=user, message=message, message_type=message_type)


@app.route('/users')
def users():
    conn = get_db()
    all_users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    return render_page('All Users', USERS_CONTENT, users=all_users)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)