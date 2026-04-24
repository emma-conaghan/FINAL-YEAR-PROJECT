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
        nav { background: #333; padding: 10px 20px; margin-bottom: 20px; border-radius: 5px; }
        nav a { color: white; text-decoration: none; margin-right: 20px; font-weight: bold; }
        nav a:hover { color: #ddd; }
        .container { background: white; padding: 30px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        label { display: block; margin-top: 15px; font-weight: bold; color: #555; }
        input, textarea { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        textarea { height: 80px; resize: vertical; }
        button, .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; margin-top: 20px; }
        button:hover, .btn:hover { background: #0056b3; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #1e7e34; }
        .profile-detail { padding: 10px 0; border-bottom: 1px solid #eee; }
        .profile-detail strong { display: inline-block; width: 120px; color: #555; }
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
<h1>User Profile Management</h1>
<p>Welcome to the User Profile Management System. You can:</p>
<ul>
    <li><a href="/create">Create a new account</a></li>
    <li><a href="/view">View a profile by Account ID</a></li>
</ul>
<h2>All Accounts</h2>
{% if users %}
<table style="width:100%; border-collapse: collapse;">
    <tr style="background: #333; color: white;">
        <th style="padding: 10px; text-align: left;">ID</th>
        <th style="padding: 10px; text-align: left;">Name</th>
        <th style="padding: 10px; text-align: left;">Email</th>
        <th style="padding: 10px; text-align: left;">Actions</th>
    </tr>
    {% for user in users %}
    <tr style="border-bottom: 1px solid #ddd;">
        <td style="padding: 10px;">{{ user['id'] }}</td>
        <td style="padding: 10px;">{{ user['name'] }}</td>
        <td style="padding: 10px;">{{ user['email'] }}</td>
        <td style="padding: 10px;">
            <a href="/view/{{ user['id'] }}">View</a> |
            <a href="/edit/{{ user['id'] }}">Edit</a>
        </td>
    </tr>
    {% endfor %}
</table>
{% else %}
<p>No accounts yet. <a href="/create">Create one!</a></p>
{% endif %}
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Create Account</h1>
{% if message %}
<div class="message message-success">{{ message }} Your Account ID is <strong>{{ account_id }}</strong>.</div>
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
    <input type="text" id="phone" name="phone" placeholder="Enter your phone number">
    
    <label for="address">Address</label>
    <textarea id="address" name="address" placeholder="Enter your address"></textarea>
    
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
        <input type="number" id="account_id" name="account_id" required placeholder="e.g. 1" value="{{ search_id or '' }}">
    </div>
    <button type="submit">Search</button>
</form>
{% if error %}
<div class="message message-error" style="margin-top: 20px;">{{ error }}</div>
{% endif %}
{% if user %}
<div style="margin-top: 30px;">
    <h2>Profile Details</h2>
    <div class="profile-detail"><strong>Account ID:</strong> {{ user['id'] }}</div>
    <div class="profile-detail"><strong>Name:</strong> {{ user['name'] }}</div>
    <div class="profile-detail"><strong>Email:</strong> {{ user['email'] }}</div>
    <div class="profile-detail"><strong>Phone:</strong> {{ user['phone'] or 'Not provided' }}</div>
    <div class="profile-detail"><strong>Address:</strong> {{ user['address'] or 'Not provided' }}</div>
    <a href="/edit/{{ user['id'] }}" class="btn btn-success" style="margin-top: 20px;">Edit Profile</a>
</div>
{% endif %}
{% endblock %}
'''

VIEW_PROFILE_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Profile Details</h1>
{% if user %}
<div>
    <div class="profile-detail"><strong>Account ID:</strong> {{ user['id'] }}</div>
    <div class="profile-detail"><strong>Name:</strong> {{ user['name'] }}</div>
    <div class="profile-detail"><strong>Email:</strong> {{ user['email'] }}</div>
    <div class="profile-detail"><strong>Phone:</strong> {{ user['phone'] or 'Not provided' }}</div>
    <div class="profile-detail"><strong>Address:</strong> {{ user['address'] or 'Not provided' }}</div>
    <a href="/edit/{{ user['id'] }}" class="btn btn-success" style="margin-top: 20px;">Edit Profile</a>
</div>
{% else %}
<div class="message message-error">Profile not found.</div>
{% endif %}
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Edit Profile (Account #{{ user['id'] }})</h1>
{% if message %}
<div class="message message-success">{{ message }}</div>
{% endif %}
{% if error %}
<div class="message message-error">{{ error }}</div>
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
    <a href="/view/{{ user['id'] }}" class="btn" style="background: #6c757d; margin-left: 10px;">Cancel</a>
</form>
{% endblock %}
'''

@app.route('/')
def home():
    conn = get_db()
    users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    return render_template_string(HOME_TEMPLATE, base=BASE_TEMPLATE, title='Home - User Profiles', users=users)

@app.route('/create', methods=['GET', 'POST'])
def create():
    message = None
    error = None
    account_id = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            error = 'Name and Email are required.'
        else:
            conn = get_db()
            cursor = conn.execute(
                'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
                (name, email, phone, address)
            )
            conn.commit()
            account_id = cursor.lastrowid
            conn.close()
            message = 'Account created successfully!'
    return render_template_string(CREATE_TEMPLATE, base=BASE_TEMPLATE, title='Create Account', message=message, error=error, account_id=account_id)

@app.route('/view')
def view_search():
    user = None
    error = None
    search_id = request.args.get('account_id')
    if search_id:
        try:
            aid = int(search_id)
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (aid,)).fetchone()
            conn.close()
            if not user:
                error = f'No account found with ID {aid}.'
        except ValueError:
            error = 'Please enter a valid numeric Account ID.'
    return render_template_string(VIEW_SEARCH_TEMPLATE, base=BASE_TEMPLATE, title='View Profile', user=user, error=error, search_id=search_id)

@app.route('/view/<int:account_id>')
def view_profile(account_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()
    return render_template_string(VIEW_PROFILE_TEMPLATE, base=BASE_TEMPLATE, title='View Profile', user=user)

@app.route('/edit/<int:account_id>', methods=['GET', 'POST'])
def edit_profile(account_id):
    conn = get_db()
    message = None
    error = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            error = 'Name and Email are required.'
        else:
            conn.execute(
                'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
                (name, email, phone, address, account_id)
            )
            conn.commit()
            message = 'Profile updated successfully!'
    user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    conn.close()
    if not user:
        return redirect(url_for('view_search'))
    return render_template_string(EDIT_TEMPLATE, base=BASE_TEMPLATE, title='Edit Profile', user=user, message=message, error=error)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)