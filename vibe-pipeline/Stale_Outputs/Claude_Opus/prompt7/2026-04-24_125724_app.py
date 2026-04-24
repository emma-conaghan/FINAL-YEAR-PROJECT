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
<html>
<head>
    <title>{{ title }}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f4f4f4; color: #333; }
        nav { background: #2c3e50; padding: 15px 30px; }
        nav a { color: white; text-decoration: none; margin-right: 20px; font-size: 16px; }
        nav a:hover { text-decoration: underline; }
        .container { max-width: 700px; margin: 30px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { margin-bottom: 20px; color: #2c3e50; }
        label { display: block; margin-top: 15px; font-weight: bold; }
        input, textarea { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        textarea { height: 80px; resize: vertical; }
        button { margin-top: 20px; padding: 12px 30px; background: #2c3e50; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background: #34495e; }
        .message { padding: 10px 15px; margin-bottom: 15px; border-radius: 4px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .profile-field { margin-bottom: 12px; }
        .profile-field strong { display: inline-block; width: 120px; }
        .search-form { display: flex; gap: 10px; align-items: end; }
        .search-form input { flex: 1; }
        .search-form button { margin-top: 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        a.btn { display: inline-block; padding: 6px 15px; background: #3498db; color: white; text-decoration: none; border-radius: 4px; font-size: 13px; }
        a.btn:hover { background: #2980b9; }
        a.btn-edit { background: #e67e22; }
        a.btn-edit:hover { background: #d35400; }
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
<h1>Welcome to Profile Manager</h1>
<p style="margin-bottom:20px;">Manage user profiles easily. Choose an option below:</p>
<p><a class="btn" href="/create">Create New Account</a></p>
<br>
<p><a class="btn" href="/view">View a Profile by ID</a></p>
<br>
<p><a class="btn" href="/accounts">View All Accounts</a></p>
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
    <input type="text" id="name" name="name" required value="{{ values.get('name', '') }}">
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required value="{{ values.get('email', '') }}">
    
    <label for="phone">Phone Number</label>
    <input type="text" id="phone" name="phone" value="{{ values.get('phone', '') }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ values.get('address', '') }}</textarea>
    
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
    <div class="search-form">
        <div style="flex:1;">
            <label for="account_id">Enter Account ID</label>
            <input type="number" id="account_id" name="id" min="1" value="{{ search_id or '' }}" placeholder="e.g. 1">
        </div>
        <button type="submit">Search</button>
    </div>
</form>
{% if user %}
<hr style="margin: 20px 0;">
<h2 style="margin-bottom: 15px;">Profile Details</h2>
<div class="profile-field"><strong>Account ID:</strong> {{ user['id'] }}</div>
<div class="profile-field"><strong>Name:</strong> {{ user['name'] }}</div>
<div class="profile-field"><strong>Email:</strong> {{ user['email'] }}</div>
<div class="profile-field"><strong>Phone:</strong> {{ user['phone'] or 'N/A' }}</div>
<div class="profile-field"><strong>Address:</strong> {{ user['address'] or 'N/A' }}</div>
<div class="profile-field"><strong>Created:</strong> {{ user['created_at'] }}</div>
<br>
<a class="btn btn-edit" href="/edit/{{ user['id'] }}">Edit Profile</a>
{% endif %}
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Edit Profile (ID: {{ user['id'] }})</h1>
{% if message %}
<div class="message {{ message_type }}">{{ message }}</div>
{% endif %}
<form method="POST" action="/edit/{{ user['id'] }}">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required value="{{ user['name'] }}">
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required value="{{ user['email'] }}">
    
    <label for="phone">Phone Number</label>
    <input type="text" id="phone" name="phone" value="{{ user['phone'] or '' }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ user['address'] or '' }}</textarea>
    
    <button type="submit">Update Profile</button>
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
        <td>{{ user['id'] }}</td>
        <td>{{ user['name'] }}</td>
        <td>{{ user['email'] }}</td>
        <td>{{ user['phone'] or 'N/A' }}</td>
        <td>
            <a class="btn" href="/view?id={{ user['id'] }}">View</a>
            <a class="btn btn-edit" href="/edit/{{ user['id'] }}">Edit</a>
        </td>
    </tr>
    {% endfor %}
</table>
{% else %}
<p>No accounts found. <a href="/create">Create one</a>.</p>
{% endif %}
{% endblock %}
'''

def render(template_str, **kwargs):
    full_template = template_str.replace('{% extends "base" %}', BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '{% block content %}{% endblock %}'))
    combined = BASE_TEMPLATE.replace('{% block content %}{% endblock %}',
        template_str.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''))
    return render_template_string(combined, **kwargs)


@app.route('/')
def home():
    return render(HOME_TEMPLATE, title='Profile Manager')


@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            return render(CREATE_TEMPLATE, title='Create Account',
                          message='Name and Email are required.', message_type='error',
                          values={'name': name, 'email': email, 'phone': phone, 'address': address})

        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
            (name, email, phone, address)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        return render(CREATE_TEMPLATE, title='Create Account',
                      message=f'Account created successfully! Your Account ID is {user_id}.',
                      message_type='success', values={})

    return render(CREATE_TEMPLATE, title='Create Account', message=None, message_type='', values={})


@app.route('/view')
def view():
    account_id = request.args.get('id', '').strip()
    user = None
    message = None
    message_type = ''

    if account_id:
        try:
            aid = int(account_id)
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (aid,)).fetchone()
            conn.close()
            if not user:
                message = f'No account found with ID {aid}.'
                message_type = 'error'
        except ValueError:
            message = 'Please enter a valid numeric ID.'
            message_type = 'error'

    return render(VIEW_TEMPLATE, title='View Profile', user=user, search_id=account_id,
                  message=message, message_type=message_type)


@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit(user_id):
    conn = get_db()

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            user = {'id': user_id, 'name': name, 'email': email, 'phone': phone, 'address': address}
            conn.close()
            return render(EDIT_TEMPLATE, title='Edit Profile', user=user,
                          message='Name and Email are required.', message_type='error')

        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()

        if not user:
            return redirect(url_for('view'))

        return render(EDIT_TEMPLATE, title='Edit Profile', user=user,
                      message='Profile updated successfully!', message_type='success')

    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        return redirect(url_for('view'))

    return render(EDIT_TEMPLATE, title='Edit Profile', user=user, message=None, message_type='')


@app.route('/accounts')
def accounts():
    conn = get_db()
    users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    return render(ACCOUNTS_TEMPLATE, title='All Accounts', users=users)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)