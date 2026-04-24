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
        nav a { color: white; text-decoration: none; margin-right: 20px; font-weight: bold; }
        nav a:hover { color: #aaa; }
        .container { background: white; padding: 30px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        label { display: block; margin-top: 15px; font-weight: bold; color: #555; }
        input, textarea { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        textarea { height: 80px; resize: vertical; }
        button, .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; margin-top: 20px; }
        button:hover, .btn:hover { background: #0056b3; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #1e7e34; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .profile-field { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .profile-field strong { color: #333; }
        .message { padding: 15px; margin-bottom: 20px; border-radius: 4px; }
        .message-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: bold; }
        tr:hover { background: #f1f1f1; }
        .search-form { display: flex; gap: 10px; align-items: end; }
        .search-form input { margin-top: 0; }
        .search-form button { margin-top: 0; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/create">Create Account</a>
        <a href="/view">View Profile</a>
        <a href="/accounts">All Accounts</a>
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
<h1>Welcome to User Profile Manager</h1>
<p>Manage your user profiles with ease.</p>
<div style="margin-top: 30px;">
    <a href="/create" class="btn btn-success">Create New Account</a>
    <a href="/view" class="btn">View a Profile</a>
    <a href="/accounts" class="btn" style="background: #6c757d;">All Accounts</a>
</div>
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Create Account</h1>
{% if message %}
<div class="message message-success">{{ message }}</div>
{% endif %}
{% if error %}
<div class="message message-error">{{ error }}</div>
{% endif %}
<form method="POST" action="/create">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required placeholder="Enter your full name">
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required placeholder="Enter your email address">
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone" placeholder="Enter your phone number">
    
    <label for="address">Address</label>
    <textarea id="address" name="address" placeholder="Enter your address"></textarea>
    
    <button type="submit">Create Account</button>
</form>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>View Profile</h1>
<form method="GET" action="/view" class="search-form">
    <div style="flex-grow: 1;">
        <label for="account_id" style="margin-top: 0;">Account ID</label>
        <input type="number" id="account_id" name="id" placeholder="Enter account ID" value="{{ search_id or '' }}" min="1">
    </div>
    <button type="submit">Search</button>
</form>

{% if error %}
<div class="message message-error" style="margin-top: 20px;">{{ error }}</div>
{% endif %}

{% if user %}
<div style="margin-top: 25px;">
    <h2>Profile Details</h2>
    <div class="profile-field"><strong>Account ID:</strong> {{ user.id }}</div>
    <div class="profile-field"><strong>Name:</strong> {{ user.name }}</div>
    <div class="profile-field"><strong>Email:</strong> {{ user.email }}</div>
    <div class="profile-field"><strong>Phone:</strong> {{ user.phone or 'Not provided' }}</div>
    <div class="profile-field"><strong>Address:</strong> {{ user.address or 'Not provided' }}</div>
    <a href="/edit/{{ user.id }}" class="btn btn-success">Edit Profile</a>
</div>
{% endif %}
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Edit Profile - Account #{{ user.id }}</h1>
{% if message %}
<div class="message message-success">{{ message }}</div>
{% endif %}
{% if error %}
<div class="message message-error">{{ error }}</div>
{% endif %}
<form method="POST" action="/edit/{{ user.id }}">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required value="{{ user.name }}">
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required value="{{ user.email }}">
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone" value="{{ user.phone or '' }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ user.address or '' }}</textarea>
    
    <button type="submit">Update Profile</button>
    <a href="/view?id={{ user.id }}" class="btn" style="background: #6c757d; margin-left: 10px;">Cancel</a>
</form>
{% endblock %}
'''

ACCOUNTS_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>All Accounts</h1>
{% if users %}
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Email</th>
            <th>Phone</th>
            <th>Actions</th>
        </tr>
    </thead>
    <tbody>
        {% for user in users %}
        <tr>
            <td>{{ user.id }}</td>
            <td>{{ user.name }}</td>
            <td>{{ user.email }}</td>
            <td>{{ user.phone or 'N/A' }}</td>
            <td>
                <a href="/view?id={{ user.id }}" style="color: #007bff; margin-right: 10px;">View</a>
                <a href="/edit/{{ user.id }}" style="color: #28a745;">Edit</a>
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No accounts found. <a href="/create">Create one now!</a></p>
{% endif %}
{% endblock %}
'''

def render(template_str, **kwargs):
    from jinja2 import Environment, BaseLoader
    env = Environment(loader=BaseLoader())
    base_tmpl = env.from_string(BASE_TEMPLATE)
    env.globals['base'] = base_tmpl

    class CustomLoader:
        pass

    full_template = template_str.replace('{% extends "base" %}', '')
    block_content = full_template.replace('{% block content %}', '').replace('{% endblock %}', '')
    final = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', block_content)
    return render_template_string(final, **kwargs)


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
            return render(CREATE_TEMPLATE, title='Create Account', error='Name and email are required.', message=None)

        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
            (name, email, phone, address)
        )
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()

        return render(CREATE_TEMPLATE, title='Create Account',
                      message=f'Account created successfully! Your Account ID is {new_id}.',
                      error=None)

    return render(CREATE_TEMPLATE, title='Create Account', message=None, error=None)


@app.route('/view')
def view():
    account_id = request.args.get('id')
    if not account_id:
        return render(VIEW_TEMPLATE, title='View Profile', user=None, error=None, search_id=None)

    try:
        account_id = int(account_id)
    except ValueError:
        return render(VIEW_TEMPLATE, title='View Profile', user=None,
                      error='Please enter a valid numeric account ID.', search_id=account_id)

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()

    if not user:
        return render(VIEW_TEMPLATE, title='View Profile', user=None,
                      error=f'No account found with ID {account_id}.', search_id=account_id)

    return render(VIEW_TEMPLATE, title='View Profile', user=user, error=None, search_id=account_id)


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
            return render(EDIT_TEMPLATE, title='Edit Profile', user=user,
                          error='Name and email are required.', message=None)

        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()

        return render(EDIT_TEMPLATE, title='Edit Profile', user=user,
                      message='Profile updated successfully!', error=None)

    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        return redirect(url_for('view'))

    return render(EDIT_TEMPLATE, title='Edit Profile', user=user, message=None, error=None)


@app.route('/accounts')
def accounts():
    conn = get_db()
    users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    return render(ACCOUNTS_TEMPLATE, title='All Accounts', users=users)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)