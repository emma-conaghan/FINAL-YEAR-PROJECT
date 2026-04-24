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
        nav a:hover { color: #4CAF50; }
        .container { background: white; padding: 30px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        label { display: block; margin-top: 15px; font-weight: bold; color: #555; }
        input, textarea { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        textarea { height: 80px; resize: vertical; }
        button, .btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; margin-top: 20px; }
        button:hover, .btn:hover { background: #45a049; }
        .btn-blue { background: #2196F3; }
        .btn-blue:hover { background: #1976D2; }
        .btn-red { background: #f44336; }
        .btn-red:hover { background: #d32f2f; }
        .profile-card { border: 1px solid #ddd; padding: 20px; border-radius: 5px; margin-top: 15px; }
        .profile-card p { margin: 10px 0; }
        .profile-card strong { color: #333; min-width: 100px; display: inline-block; }
        .message { padding: 10px 20px; border-radius: 4px; margin-bottom: 15px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4CAF50; color: white; }
        tr:hover { background: #f5f5f5; }
        .search-box { display: flex; gap: 10px; align-items: end; }
        .search-box input { flex: 1; }
        .search-box button { margin-top: 0; }
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
<p>Manage your user profiles easily.</p>
<div style="margin-top: 30px;">
    <a href="/create" class="btn">Create New Account</a>
    <a href="/view" class="btn btn-blue" style="margin-left: 10px;">View a Profile</a>
    <a href="/accounts" class="btn btn-blue" style="margin-left: 10px;">All Accounts</a>
</div>
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
    <input type="text" id="phone" name="phone">

    <label for="address">Address</label>
    <textarea id="address" name="address"></textarea>

    <button type="submit">Create Account</button>
</form>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>View Profile</h1>
{% if message %}
<div class="message {{ message_type }}">{{ message }}</div>
{% endif %}
<form method="GET" action="/view">
    <div class="search-box">
        <div style="flex:1;">
            <label for="account_id">Enter Account ID</label>
            <input type="number" id="account_id" name="account_id" value="{{ account_id or '' }}" required min="1">
        </div>
        <button type="submit">Look Up</button>
    </div>
</form>
{% if user %}
<div class="profile-card">
    <h2>Profile Details</h2>
    <p><strong>Account ID:</strong> {{ user['id'] }}</p>
    <p><strong>Name:</strong> {{ user['name'] }}</p>
    <p><strong>Email:</strong> {{ user['email'] }}</p>
    <p><strong>Phone:</strong> {{ user['phone'] or 'Not provided' }}</p>
    <p><strong>Address:</strong> {{ user['address'] or 'Not provided' }}</p>
    <a href="/edit/{{ user['id'] }}" class="btn btn-blue">Edit Profile</a>
</div>
{% endif %}
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Edit Profile - Account #{{ user['id'] }}</h1>
{% if message %}
<div class="message {{ message_type }}">{{ message }}</div>
{% endif %}
<form method="POST" action="/edit/{{ user['id'] }}">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" value="{{ user['name'] }}" required>

    <label for="email">Email *</label>
    <input type="email" id="email" name="email" value="{{ user['email'] }}" required>

    <label for="phone">Phone Number</label>
    <input type="text" id="phone" name="phone" value="{{ user['phone'] or '' }}">

    <label for="address">Address</label>
    <textarea id="address" name="address">{{ user['address'] or '' }}</textarea>

    <button type="submit">Update Profile</button>
    <a href="/view?account_id={{ user['id'] }}" class="btn btn-blue" style="margin-left: 10px;">Cancel</a>
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
            <td>{{ user['id'] }}</td>
            <td>{{ user['name'] }}</td>
            <td>{{ user['email'] }}</td>
            <td>{{ user['phone'] or 'N/A' }}</td>
            <td>
                <a href="/view?account_id={{ user['id'] }}">View</a> |
                <a href="/edit/{{ user['id'] }}">Edit</a>
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

def render(template_string, **kwargs):
    from jinja2 import Environment, BaseLoader, DictLoader
    templates = {
        'base': BASE_TEMPLATE,
        'page': template_string
    }
    env = Environment(loader=DictLoader(templates))
    template = env.from_string('{% extends "base" %}' + template_string.split('{% extends "base" %}')[1] if '{% extends "base" %}' in template_string else template_string)
    return template.render(**kwargs)

@app.route('/')
def home():
    return render_template_string(BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
    <h1>Welcome to User Profile Manager</h1>
    <p>Manage your user profiles easily.</p>
    <div style="margin-top: 30px;">
        <a href="/create" class="btn">Create New Account</a>
        <a href="/view" class="btn btn-blue" style="margin-left: 10px;">View a Profile</a>
        <a href="/accounts" class="btn btn-blue" style="margin-left: 10px;">All Accounts</a>
    </div>
    '''), title='Home')

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
            return redirect(url_for('view_profile', account_id=user_id, created='1'))

    content = '''
    <h1>Create Account</h1>
    ''' + (('<div class="message ' + (message_type or '') + '">' + (message or '') + '</div>') if message else '') + '''
    <form method="POST" action="/create">
        <label for="name">Name *</label>
        <input type="text" id="name" name="name" required>
        <label for="email">Email *</label>
        <input type="email" id="email" name="email" required>
        <label for="phone">Phone Number</label>
        <input type="text" id="phone" name="phone">
        <label for="address">Address</label>
        <textarea id="address" name="address"></textarea>
        <button type="submit">Create Account</button>
    </form>
    '''
    return render_template_string(
        BASE_TEMPLATE.replace('{% block content %}{% endblock %}', content),
        title='Create Account'
    )

@app.route('/view')
def view_profile():
    account_id = request.args.get('account_id', '').strip()
    created = request.args.get('created', '')
    user = None
    message = None
    message_type = None

    if created == '1' and account_id:
        message = f'Account #{account_id} created successfully!'
        message_type = 'success'

    if account_id:
        try:
            aid = int(account_id)
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (aid,)).fetchone()
            conn.close()
            if not user and not message:
                message = f'No account found with ID {aid}.'
                message_type = 'error'
        except ValueError:
            message = 'Please enter a valid numeric account ID.'
            message_type = 'error'

    user_html = ''
    if user:
        user_html = f'''
        <div class="profile-card">
            <h2>Profile Details</h2>
            <p><strong>Account ID:</strong> {user['id']}</p>
            <p><strong>Name:</strong> {user['name']}</p>
            <p><strong>Email:</strong> {user['email']}</p>
            <p><strong>Phone:</strong> {user['phone'] or 'Not provided'}</p>
            <p><strong>Address:</strong> {user['address'] or 'Not provided'}</p>
            <a href="/edit/{user['id']}" class="btn btn-blue">Edit Profile</a>
        </div>
        '''

    message_html = ''
    if message:
        message_html = f'<div class="message {message_type}">{message}</div>'

    content = f'''
    <h1>View Profile</h1>
    {message_html}
    <form method="GET" action="/view">
        <div class="search-box">
            <div style="flex:1;">
                <label for="account_id">Enter Account ID</label>
                <input type="number" id="account_id" name="account_id" value="{account_id}" required min="1">
            </div>
            <button type="submit">Look Up</button>
        </div>
    </form>
    {user_html}
    '''
    return render_template_string(
        BASE_TEMPLATE.replace('{% block content %}{% endblock %}', content),
        title='View Profile'
    )

@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_profile(user_id):
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
            message = 'Profile