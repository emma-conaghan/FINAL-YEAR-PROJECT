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
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone TEXT,
        address TEXT
    )''')
    conn.commit()
    conn.close()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        nav { background: #333; padding: 10px 20px; border-radius: 5px; margin-bottom: 20px; }
        nav a { color: white; text-decoration: none; margin-right: 20px; font-size: 16px; }
        nav a:hover { text-decoration: underline; }
        .container { background: white; padding: 30px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        label { display: block; margin-top: 15px; font-weight: bold; color: #555; }
        input, textarea { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        textarea { height: 80px; resize: vertical; }
        button, .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; margin-top: 20px; }
        button:hover, .btn:hover { background: #0056b3; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #1e7e34; }
        .profile-field { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .profile-field strong { color: #333; }
        .message { padding: 15px; margin-bottom: 20px; border-radius: 4px; }
        .message-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .search-form { display: flex; gap: 10px; align-items: end; }
        .search-form input { flex: 1; }
        .search-form button { margin-top: 0; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/create">Create Account</a>
        <a href="/view">View Profile</a>
    </nav>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

HOME_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Welcome to Profile Manager</h1>
<p>Manage your user profiles easily.</p>
<ul style="list-style: none; padding: 0;">
    <li style="margin: 15px 0;"><a class="btn" href="/create">Create New Account</a></li>
    <li style="margin: 15px 0;"><a class="btn btn-success" href="/view">View Profile by ID</a></li>
</ul>
{% if users %}
<h2>All Accounts</h2>
{% for user in users %}
<div class="profile-field">
    <strong>ID {{ user.id }}:</strong> {{ user.name }} ({{ user.email }})
    <a href="/view/{{ user.id }}" style="margin-left: 10px;">View</a>
    <a href="/edit/{{ user.id }}" style="margin-left: 10px;">Edit</a>
</div>
{% endfor %}
{% endif %}
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Create Account</h1>
{% if message %}
<div class="message message-success">{{ message }} <a href="/view/{{ new_id }}">View Profile</a></div>
{% endif %}
<form method="POST">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required>
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required>
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone">
    
    <label for="address">Address</label>
    <textarea id="address" name="address"></textarea>
    
    <button type="submit">Create Account</button>
</form>
{% endblock %}
'''

VIEW_SEARCH_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>View Profile</h1>
<form method="GET" action="/view" class="search-form">
    <div style="flex:1;">
        <label for="account_id">Enter Account ID</label>
        <input type="number" id="account_id" name="id" min="1" required value="{{ search_id or '' }}">
    </div>
    <button type="submit">Search</button>
</form>
{% if error %}
<div class="message message-error" style="margin-top: 20px;">{{ error }}</div>
{% endif %}
{% if user %}
<h2 style="margin-top: 30px;">Profile Details</h2>
<div class="profile-field"><strong>Account ID:</strong> {{ user.id }}</div>
<div class="profile-field"><strong>Name:</strong> {{ user.name }}</div>
<div class="profile-field"><strong>Email:</strong> {{ user.email }}</div>
<div class="profile-field"><strong>Phone:</strong> {{ user.phone or 'Not provided' }}</div>
<div class="profile-field"><strong>Address:</strong> {{ user.address or 'Not provided' }}</div>
<a class="btn" href="/edit/{{ user.id }}">Edit Profile</a>
{% endif %}
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Profile Details</h1>
{% if user %}
<div class="profile-field"><strong>Account ID:</strong> {{ user.id }}</div>
<div class="profile-field"><strong>Name:</strong> {{ user.name }}</div>
<div class="profile-field"><strong>Email:</strong> {{ user.email }}</div>
<div class="profile-field"><strong>Phone:</strong> {{ user.phone or 'Not provided' }}</div>
<div class="profile-field"><strong>Address:</strong> {{ user.address or 'Not provided' }}</div>
<a class="btn" href="/edit/{{ user.id }}">Edit Profile</a>
{% else %}
<div class="message message-error">User not found.</div>
{% endif %}
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Edit Profile</h1>
{% if message %}
<div class="message message-success">{{ message }}</div>
{% endif %}
{% if error %}
<div class="message message-error">{{ error }}</div>
{% endif %}
{% if user %}
<form method="POST">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required value="{{ user.name }}">
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required value="{{ user.email }}">
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone" value="{{ user.phone or '' }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ user.address or '' }}</textarea>
    
    <button type="submit">Update Profile</button>
</form>
<a href="/view/{{ user.id }}" style="display: inline-block; margin-top: 10px;">Back to Profile</a>
{% else %}
<div class="message message-error">User not found.</div>
{% endif %}
{% endblock %}
'''


@app.route('/')
def home():
    conn = get_db()
    users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    return render_template_string(HOME_TEMPLATE, base=BASE_TEMPLATE, title='Home - Profile Manager', users=users)


@app.route('/create', methods=['GET', 'POST'])
def create():
    message = None
    new_id = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        
        if name and email:
            conn = get_db()
            cursor = conn.execute(
                'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
                (name, email, phone, address)
            )
            conn.commit()
            new_id = cursor.lastrowid
            conn.close()
            message = f'Account created successfully! Your Account ID is {new_id}.'
    
    return render_template_string(CREATE_TEMPLATE, base=BASE_TEMPLATE, title='Create Account', message=message, new_id=new_id)


@app.route('/view')
def view_search():
    user = None
    error = None
    search_id = request.args.get('id')
    
    if search_id:
        try:
            account_id = int(search_id)
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
            conn.close()
            if not user:
                error = f'No account found with ID {account_id}.'
        except ValueError:
            error = 'Please enter a valid numeric ID.'
    
    return render_template_string(VIEW_SEARCH_TEMPLATE, base=BASE_TEMPLATE, title='View Profile', user=user, error=error, search_id=search_id)


@app.route('/view/<int:user_id>')
def view_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return render_template_string(VIEW_TEMPLATE, base=BASE_TEMPLATE, title='Profile Details', user=user)


@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_profile(user_id):
    conn = get_db()
    message = None
    error = None
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        
        if name and email:
            conn.execute(
                'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
                (name, email, phone, address, user_id)
            )
            conn.commit()
            message = 'Profile updated successfully!'
        else:
            error = 'Name and email are required.'
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return render_template_string(EDIT_TEMPLATE, base=BASE_TEMPLATE, title='Edit Profile', user=user, message=message, error=error)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)