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
        nav a:hover { color: #ddd; }
        .container { background: white; padding: 30px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        label { display: block; margin-top: 15px; font-weight: bold; color: #555; }
        input, textarea { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        textarea { height: 80px; resize: vertical; }
        button, .btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; margin-top: 20px; }
        button:hover, .btn:hover { background: #45a049; }
        .btn-blue { background: #2196F3; }
        .btn-blue:hover { background: #1976D2; }
        .btn-red { background: #f44336; }
        .btn-red:hover { background: #d32f2f; }
        .profile-info { margin: 10px 0; }
        .profile-info strong { display: inline-block; width: 120px; color: #555; }
        .message { padding: 15px; margin-bottom: 20px; border-radius: 4px; }
        .success { background: #dff0d8; color: #3c763d; border: 1px solid #d6e9c6; }
        .error { background: #f2dede; color: #a94442; border: 1px solid #ebccd1; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f8f8; font-weight: bold; color: #555; }
        tr:hover { background: #f5f5f5; }
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
{% extends base %}
{% block content %}
<h1>User Profile Management</h1>
<p>Welcome to the User Profile Management System. Use the navigation above to:</p>
<ul>
    <li><a href="/create">Create a new account</a></li>
    <li><a href="/view">View a profile by Account ID</a></li>
    <li><a href="/accounts">View all accounts</a></li>
</ul>
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Create Account</h1>
{% if message %}
<div class="message success">{{ message }} <a href="/view/{{ new_id }}">View Profile</a></div>
{% endif %}
{% if error %}
<div class="message error">{{ error }}</div>
{% endif %}
<form method="POST">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required value="{{ values.get('name', '') }}">
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required value="{{ values.get('email', '') }}">
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone" value="{{ values.get('phone', '') }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ values.get('address', '') }}</textarea>
    
    <button type="submit">Create Account</button>
</form>
{% endblock %}
'''

VIEW_SEARCH_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>View Profile</h1>
<form method="GET" action="/view" class="search-form">
    <div style="flex-grow:1">
        <label for="account_id" style="margin-top:0">Enter Account ID</label>
        <input type="number" id="account_id" name="id" min="1" required placeholder="e.g. 1" value="{{ search_id or '' }}">
    </div>
    <button type="submit" class="btn-blue" style="margin-top:0">Search</button>
</form>
{% if error %}
<div class="message error" style="margin-top:20px">{{ error }}</div>
{% endif %}
{% if user %}
<hr style="margin-top:25px">
<h2>Profile Details</h2>
<div class="profile-info"><strong>Account ID:</strong> {{ user['id'] }}</div>
<div class="profile-info"><strong>Name:</strong> {{ user['name'] }}</div>
<div class="profile-info"><strong>Email:</strong> {{ user['email'] }}</div>
<div class="profile-info"><strong>Phone:</strong> {{ user['phone'] or 'N/A' }}</div>
<div class="profile-info"><strong>Address:</strong> {{ user['address'] or 'N/A' }}</div>
<a href="/edit/{{ user['id'] }}" class="btn btn-blue">Edit Profile</a>
{% endif %}
{% endblock %}
'''

VIEW_PROFILE_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Profile Details</h1>
{% if message %}
<div class="message success">{{ message }}</div>
{% endif %}
{% if user %}
<div class="profile-info"><strong>Account ID:</strong> {{ user['id'] }}</div>
<div class="profile-info"><strong>Name:</strong> {{ user['name'] }}</div>
<div class="profile-info"><strong>Email:</strong> {{ user['email'] }}</div>
<div class="profile-info"><strong>Phone:</strong> {{ user['phone'] or 'N/A' }}</div>
<div class="profile-info"><strong>Address:</strong> {{ user['address'] or 'N/A' }}</div>
<a href="/edit/{{ user['id'] }}" class="btn btn-blue">Edit Profile</a>
{% else %}
<div class="message error">Account not found.</div>
{% endif %}
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Edit Profile - Account #{{ user['id'] }}</h1>
{% if message %}
<div class="message success">{{ message }}</div>
{% endif %}
{% if error %}
<div class="message error">{{ error }}</div>
{% endif %}
<form method="POST">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required value="{{ values.get('name', user['name']) }}">
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required value="{{ values.get('email', user['email']) }}">
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone" value="{{ values.get('phone', user['phone'] or '') }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ values.get('address', user['address'] or '') }}</textarea>
    
    <button type="submit">Update Profile</button>
    <a href="/view/{{ user['id'] }}" class="btn btn-blue" style="margin-left:10px">Cancel</a>
</form>
{% endblock %}
'''

ACCOUNTS_TEMPLATE = '''
{% extends base %}
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
                <a href="/view/{{ user['id'] }}">View</a> |
                <a href="/edit/{{ user['id'] }}">Edit</a>
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No accounts found. <a href="/create">Create one now</a>.</p>
{% endif %}
{% endblock %}
'''

def render(template_string, **kwargs):
    kwargs['base'] = BASE_TEMPLATE
    return render_template_string(template_string, **kwargs)


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
            return render(CREATE_TEMPLATE, title='Create Account',
                          error='Name and email are required.',
                          values=request.form)

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
                      new_id=new_id, values={})

    return render(CREATE_TEMPLATE, title='Create Account', values={})


@app.route('/view')
def view_search():
    account_id = request.args.get('id')
    user = None
    error = None

    if account_id:
        try:
            aid = int(account_id)
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (aid,)).fetchone()
            conn.close()
            if not user:
                error = f'No account found with ID {aid}.'
        except ValueError:
            error = 'Please enter a valid numeric Account ID.'

    return render(VIEW_SEARCH_TEMPLATE, title='View Profile',
                  user=user, error=error, search_id=account_id)


@app.route('/view/<int:account_id>')
def view_profile(account_id):
    message = request.args.get('message', '')
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()
    return render(VIEW_PROFILE_TEMPLATE, title='View Profile', user=user, message=message)


@app.route('/edit/<int:account_id>', methods=['GET', 'POST'])
def edit_profile(account_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()

    if not user:
        conn.close()
        return redirect(url_for('view_search'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            conn.close()
            return render(EDIT_TEMPLATE, title='Edit Profile',
                          user=user, error='Name and email are required.',
                          values=request.form)

        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, account_id)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
        conn.close()

        return render(EDIT_TEMPLATE, title='Edit Profile',
                      user=user, message='Profile updated successfully!', values={})

    conn.close()
    return render(EDIT_TEMPLATE, title='Edit Profile', user=user, values={})


@app.route('/accounts')
def accounts():
    conn = get_db()
    users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    return render(ACCOUNTS_TEMPLATE, title='All Accounts', users=users)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)