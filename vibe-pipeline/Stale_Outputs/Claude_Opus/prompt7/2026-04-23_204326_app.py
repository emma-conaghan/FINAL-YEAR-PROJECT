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
        button, .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 20px; display: inline-block; text-decoration: none; }
        button:hover, .btn:hover { background: #0056b3; }
        .btn-secondary { background: #6c757d; }
        .btn-secondary:hover { background: #545b62; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .profile-field { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .profile-field strong { color: #333; }
        .message { padding: 15px; margin-bottom: 20px; border-radius: 4px; }
        .message-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .search-form { display: flex; gap: 10px; align-items: end; }
        .search-form input { flex: 1; }
        .search-form button { margin-top: 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: bold; }
        tr:hover { background: #f1f1f1; }
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
<ul>
    <li><a href="/create">Create a new account</a></li>
    <li><a href="/view">View a profile by Account ID</a></li>
    <li><a href="/accounts">View all accounts</a></li>
</ul>
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Create Account</h1>
{% if message %}
<div class="message message-{{ message_type }}">{{ message }}</div>
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
<div class="message message-{{ message_type }}">{{ message }}</div>
{% endif %}
<form method="GET" action="/view">
    <div class="search-form">
        <div style="flex:1">
            <label for="account_id">Enter Account ID</label>
            <input type="number" id="account_id" name="account_id" value="{{ account_id or '' }}" required min="1">
        </div>
        <button type="submit">Look Up</button>
    </div>
</form>
{% if user %}
<h2>Profile Details</h2>
<div class="profile-field"><strong>Account ID:</strong> {{ user.id }}</div>
<div class="profile-field"><strong>Name:</strong> {{ user.name }}</div>
<div class="profile-field"><strong>Email:</strong> {{ user.email }}</div>
<div class="profile-field"><strong>Phone:</strong> {{ user.phone or 'Not provided' }}</div>
<div class="profile-field"><strong>Address:</strong> {{ user.address or 'Not provided' }}</div>
<a href="/edit/{{ user.id }}" class="btn" style="margin-right:10px;">Edit Profile</a>
<a href="/delete/{{ user.id }}" class="btn btn-danger" onclick="return confirm('Are you sure you want to delete this account?');">Delete Account</a>
{% endif %}
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Edit Profile (Account #{{ user.id }})</h1>
{% if message %}
<div class="message message-{{ message_type }}">{{ message }}</div>
{% endif %}
<form method="POST" action="/edit/{{ user.id }}">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" value="{{ user.name }}" required>
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" value="{{ user.email }}" required>
    
    <label for="phone">Phone Number</label>
    <input type="text" id="phone" name="phone" value="{{ user.phone or '' }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ user.address or '' }}</textarea>
    
    <button type="submit">Update Profile</button>
    <a href="/view?account_id={{ user.id }}" class="btn btn-secondary" style="margin-left:10px;">Cancel</a>
</form>
{% endblock %}
'''

ACCOUNTS_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>All Accounts</h1>
{% if users %}
<table>
    <tr>
        <th>ID</th>
        <th>Name</th>
        <th>Email</th>
        <th>Phone</th>
        <th>Actions</th>
    </tr>
    {% for user in users %}
    <tr>
        <td>{{ user.id }}</td>
        <td>{{ user.name }}</td>
        <td>{{ user.email }}</td>
        <td>{{ user.phone or '-' }}</td>
        <td>
            <a href="/view?account_id={{ user.id }}">View</a> |
            <a href="/edit/{{ user.id }}">Edit</a>
        </td>
    </tr>
    {% endfor %}
</table>
{% else %}
<p>No accounts found. <a href="/create">Create one now</a>.</p>
{% endif %}
{% endblock %}
'''


@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE, title='Home', base=BASE_TEMPLATE)


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
            account_id = cursor.lastrowid
            conn.close()
            message = f'Account created successfully! Your Account ID is {account_id}.'
            message_type = 'success'

    return render_template_string(CREATE_TEMPLATE, title='Create Account', base=BASE_TEMPLATE,
                                 message=message, message_type=message_type)


@app.route('/view')
def view():
    account_id = request.args.get('account_id')
    user = None
    message = None
    message_type = None

    if account_id:
        try:
            account_id = int(account_id)
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
            conn.close()
            if not user:
                message = f'No account found with ID {account_id}.'
                message_type = 'error'
        except ValueError:
            message = 'Please enter a valid numeric Account ID.'
            message_type = 'error'

    return render_template_string(VIEW_TEMPLATE, title='View Profile', base=BASE_TEMPLATE,
                                 user=user, account_id=account_id, message=message, message_type=message_type)


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

    if not user:
        return redirect(url_for('view'))

    return render_template_string(EDIT_TEMPLATE, title='Edit Profile', base=BASE_TEMPLATE,
                                 user=user, message=message, message_type=message_type)


@app.route('/delete/<int:user_id>')
def delete(user_id):
    conn = get_db()
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('accounts'))


@app.route('/accounts')
def accounts():
    conn = get_db()
    users = conn.execute('SELECT * FROM users ORDER BY id').fetchall()
    conn.close()
    return render_template_string(ACCOUNTS_TEMPLATE, title='All Accounts', base=BASE_TEMPLATE, users=users)


# Override Jinja2 to support extends "base"
from jinja2 import BaseLoader, TemplateNotFound

class CustomLoader(BaseLoader):
    def get_source(self, environment, template):
        if template == 'base':
            return BASE_TEMPLATE, None, lambda: True
        raise TemplateNotFound(template)

app.jinja_loader = CustomLoader()


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)