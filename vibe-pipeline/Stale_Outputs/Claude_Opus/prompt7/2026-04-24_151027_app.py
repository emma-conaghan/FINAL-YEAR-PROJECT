from flask import Flask, Flask, request, redirect, url_for, render_template_string
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
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        nav {
            background-color: #333;
            padding: 10px 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        nav a {
            color: white;
            text-decoration: none;
            margin-right: 20px;
            font-size: 16px;
        }
        nav a:hover {
            text-decoration: underline;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
            color: #555;
        }
        input[type="text"], input[type="email"], input[type="tel"], textarea, input[type="number"] {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 14px;
        }
        textarea {
            height: 80px;
            resize: vertical;
        }
        button, input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 20px;
        }
        button:hover, input[type="submit"]:hover {
            background-color: #45a049;
        }
        .profile-card {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            margin-top: 15px;
        }
        .profile-card p {
            margin: 8px 0;
            font-size: 16px;
        }
        .profile-card strong {
            color: #333;
        }
        .message {
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .user-list {
            list-style: none;
            padding: 0;
        }
        .user-list li {
            padding: 10px 15px;
            border: 1px solid #ddd;
            margin-bottom: 5px;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .user-list li a {
            color: #4CAF50;
            text-decoration: none;
        }
        .user-list li a:hover {
            text-decoration: underline;
        }
        .btn-edit {
            background-color: #2196F3;
        }
        .btn-edit:hover {
            background-color: #1976D2;
        }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/create">Create Account</a>
        <a href="/lookup">View Profile</a>
        <a href="/users">All Users</a>
    </nav>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

HOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Welcome to User Profiles</h1>
<p>Manage your profile information easily.</p>
<ul>
    <li><a href="/create">Create a new account</a></li>
    <li><a href="/lookup">View a profile by Account ID</a></li>
    <li><a href="/users">View all users</a></li>
</ul>
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Create Account</h1>
{% if message %}
<div class="message {{ message_type }}">{{ message }}</div>
{% endif %}
<form method="POST" action="/create">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required>
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required>
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone">
    
    <label for="address">Address</label>
    <textarea id="address" name="address"></textarea>
    
    <input type="submit" value="Create Account">
</form>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Update Profile</h1>
{% if message %}
<div class="message {{ message_type }}">{{ message }}</div>
{% endif %}
<form method="POST" action="/edit/{{ user.id }}">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" value="{{ user.name }}" required>
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" value="{{ user.email }}" required>
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone" value="{{ user.phone or '' }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ user.address or '' }}</textarea>
    
    <input type="submit" value="Update Profile">
</form>
{% endblock %}
'''

PROFILE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>User Profile</h1>
{% if user %}
<div class="profile-card">
    <p><strong>Account ID:</strong> {{ user.id }}</p>
    <p><strong>Name:</strong> {{ user.name }}</p>
    <p><strong>Email:</strong> {{ user.email }}</p>
    <p><strong>Phone:</strong> {{ user.phone or 'Not provided' }}</p>
    <p><strong>Address:</strong> {{ user.address or 'Not provided' }}</p>
</div>
<a href="/edit/{{ user.id }}"><button class="btn-edit">Edit Profile</button></a>
{% else %}
<div class="message error">User not found.</div>
{% endif %}
{% endblock %}
'''

LOOKUP_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>View Profile by Account ID</h1>
{% if message %}
<div class="message error">{{ message }}</div>
{% endif %}
<form method="GET" action="/profile">
    <label for="user_id">Account ID</label>
    <input type="number" id="user_id" name="id" required min="1">
    <input type="submit" value="View Profile">
</form>
{% endblock %}
'''

USERS_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>All Users</h1>
{% if users %}
<ul class="user-list">
    {% for user in users %}
    <li>
        <span><strong>ID {{ user.id }}</strong> - {{ user.name }} ({{ user.email }})</span>
        <span>
            <a href="/profile?id={{ user.id }}">View</a> | 
            <a href="/edit/{{ user.id }}">Edit</a>
        </span>
    </li>
    {% endfor %}
</ul>
{% else %}
<p>No users found. <a href="/create">Create one!</a></p>
{% endif %}
{% endblock %}
'''

CREATED_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Account Created!</h1>
<div class="message success">Your account has been created successfully.</div>
<div class="profile-card">
    <p><strong>Account ID:</strong> {{ user.id }}</p>
    <p><strong>Name:</strong> {{ user.name }}</p>
    <p><strong>Email:</strong> {{ user.email }}</p>
    <p><strong>Phone:</strong> {{ user.phone or 'Not provided' }}</p>
    <p><strong>Address:</strong> {{ user.address or 'Not provided' }}</p>
</div>
<p>Please save your Account ID to view or update your profile later.</p>
<a href="/edit/{{ user.id }}"><button class="btn-edit">Edit Profile</button></a>
{% endblock %}
'''

def render(template_str, **kwargs):
    from jinja2 import Environment, BaseLoader
    env = Environment(loader=BaseLoader())
    base_tmpl = env.from_string(BASE_TEMPLATE)
    env.globals['base'] = base_tmpl
    
    full_template = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', template_str.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''))
    return render_template_string(full_template, **kwargs)


@app.route('/')
def home():
    return render(HOME_TEMPLATE, title='Home')


@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        
        if not name or not email:
            return render(CREATE_TEMPLATE, title='Create Account', message='Name and email are required.', message_type='error')
        
        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
            (name, email, phone, address)
        )
        conn.commit()
        user_id = cursor.lastrowid
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        
        return render(CREATED_TEMPLATE, title='Account Created', user=user)
    
    return render(CREATE_TEMPLATE, title='Create Account', message=None, message_type=None)


@app.route('/profile')
def profile():
    user_id = request.args.get('id')
    if not user_id:
        return redirect(url_for('lookup'))
    
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    return render(PROFILE_TEMPLATE, title='User Profile', user=user)


@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit(user_id):
    conn = get_db()
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        
        if not name or not email:
            user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
            conn.close()
            return render(EDIT_TEMPLATE, title='Update Profile', user=user, message='Name and email are required.', message_type='error')
        
        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        
        if user:
            return render(EDIT_TEMPLATE, title='Update Profile', user=user, message='Profile updated successfully!', message_type='success')
        return render(EDIT_TEMPLATE, title='Update Profile', user=None, message='User not found.', message_type='error')
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if not user:
        return render(PROFILE_TEMPLATE, title='User Profile', user=None)
    
    return render(EDIT_TEMPLATE, title='Update Profile', user=user, message=None, message_type=None)


@app.route('/lookup')
def lookup():
    return render(LOOKUP_TEMPLATE, title='View Profile', message=None)


@app.route('/users')
def users():
    conn = get_db()
    all_users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    return render(USERS_TEMPLATE, title='All Users', users=all_users)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)