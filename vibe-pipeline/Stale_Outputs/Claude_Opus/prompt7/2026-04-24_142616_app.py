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
        .btn-secondary { background: #6c757d; }
        .btn-secondary:hover { background: #545b62; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .profile-field { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 4px; }
        .profile-field strong { color: #333; }
        .message { padding: 15px; margin-bottom: 20px; border-radius: 4px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .search-form { display: flex; gap: 10px; align-items: end; }
        .search-form input { flex: 1; }
        .search-form button { margin-top: 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: bold; }
        tr:hover { background: #f5f5f5; }
        .id-badge { background: #007bff; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
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
<h1>Welcome to Profile Manager</h1>
<p>Manage your user profiles easily.</p>
<div style="margin-top: 30px;">
    <a href="/create" class="btn">Create New Account</a>
    <a href="/view" class="btn btn-secondary" style="margin-left: 10px;">View a Profile</a>
    <a href="/accounts" class="btn btn-secondary" style="margin-left: 10px;">All Accounts</a>
</div>
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
    <div style="flex:1">
        <label for="account_id">Enter Account ID</label>
        <input type="number" id="account_id" name="id" min="1" required placeholder="e.g. 1" value="{{ search_id or '' }}">
    </div>
    <button type="submit">Search</button>
</form>
{% if error %}
<div class="message error" style="margin-top: 20px;">{{ error }}</div>
{% endif %}
{% if user %}
<div style="margin-top: 30px;">
    <h2>Profile Details <span class="id-badge">ID: {{ user['id'] }}</span></h2>
    <div class="profile-field"><strong>Name:</strong> {{ user['name'] }}</div>
    <div class="profile-field"><strong>Email:</strong> {{ user['email'] }}</div>
    <div class="profile-field"><strong>Phone:</strong> {{ user['phone'] or 'Not provided' }}</div>
    <div class="profile-field"><strong>Address:</strong> {{ user['address'] or 'Not provided' }}</div>
    <a href="/edit/{{ user['id'] }}" class="btn" style="margin-right: 10px;">Edit Profile</a>
</div>
{% endif %}
{% endblock %}
'''

VIEW_PROFILE_TEMPLATE = '''
{% extends base %}
{% block content %}
<h2>Profile Details <span class="id-badge">ID: {{ user['id'] }}</span></h2>
<div class="profile-field"><strong>Name:</strong> {{ user['name'] }}</div>
<div class="profile-field"><strong>Email:</strong> {{ user['email'] }}</div>
<div class="profile-field"><strong>Phone:</strong> {{ user['phone'] or 'Not provided' }}</div>
<div class="profile-field"><strong>Address:</strong> {{ user['address'] or 'Not provided' }}</div>
<a href="/edit/{{ user['id'] }}" class="btn" style="margin-right: 10px;">Edit Profile</a>
<a href="/accounts" class="btn btn-secondary">Back to All Accounts</a>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends base %}
{% block content %}
<h1>Edit Profile <span class="id-badge">ID: {{ user['id'] }}</span></h1>
{% if message %}
<div class="message success">{{ message }}</div>
{% endif %}
{% if error %}
<div class="message error">{{ error }}</div>
{% endif %}
<form method="POST">
    <label for="name">Name *</label>
    <input type="text" id="name" name="name" required value="{{ user['name'] }}">
    
    <label for="email">Email *</label>
    <input type="email" id="email" name="email" required value="{{ user['email'] }}">
    
    <label for="phone">Phone Number</label>
    <input type="tel" id="phone" name="phone" value="{{ user['phone'] or '' }}">
    
    <label for="address">Address</label>
    <textarea id="address" name="address">{{ user['address'] or '' }}</textarea>
    
    <button type="submit">Update Profile</button>
    <a href="/view/{{ user['id'] }}" class="btn btn-secondary" style="margin-left: 10px;">Cancel</a>
    <a href="/delete/{{ user['id'] }}" class="btn btn-danger" style="margin-left: 10px;" onclick="return confirm('Are you sure you want to delete this account?');">Delete Account</a>
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
            <td><span class="id-badge">{{ user['id'] }}</span></td>
            <td>{{ user['name'] }}</td>
            <td>{{ user['email'] }}</td>
            <td>{{ user['phone'] or '-' }}</td>
            <td>
                <a href="/view/{{ user['id'] }}">View</a> |
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

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE, base=BASE_TEMPLATE, title='Profile Manager')

@app.route('/create', methods=['GET', 'POST'])
def create():
    message = None
    error = None
    new_id = None
    values = {}
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        values = {'name': name, 'email': email, 'phone': phone, 'address': address}
        
        if not name or not email:
            error = 'Name and email are required.'
        else:
            try:
                conn = get_db()
                cursor = conn.execute(
                    'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
                    (name, email, phone, address)
                )
                conn.commit()
                new_id = cursor.lastrowid
                conn.close()
                message = f'Account created successfully! Your Account ID is {new_id}.'
                values = {}
            except Exception as e:
                error = f'Error creating account: {str(e)}'
    
    return render_template_string(CREATE_TEMPLATE, base=BASE_TEMPLATE, title='Create Account',
                                  message=message, error=error, new_id=new_id, values=values)

@app.route('/view')
def view_search():
    search_id = request.args.get('id')
    user = None
    error = None
    
    if search_id:
        try:
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (int(search_id),)).fetchone()
            conn.close()
            if not user:
                error = f'No account found with ID {search_id}.'
        except ValueError:
            error = 'Please enter a valid numeric ID.'
        except Exception as e:
            error = f'Error: {str(e)}'
    
    return render_template_string(VIEW_SEARCH_TEMPLATE, base=BASE_TEMPLATE, title='View Profile',
                                  user=user, error=error, search_id=search_id)

@app.route('/view/<int:user_id>')
def view_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if not user:
        return render_template_string(VIEW_SEARCH_TEMPLATE, base=BASE_TEMPLATE, title='View Profile',
                                      user=None, error=f'No account found with ID {user_id}.', search_id=user_id)
    
    return render_template_string(VIEW_PROFILE_TEMPLATE, base=BASE_TEMPLATE, title=f'Profile - {user["name"]}', user=user)

@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return redirect(url_for('view_search'))
    
    message = None
    error = None
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        
        if not name or not email:
            error = 'Name and email are required.'
            user = {'id': user_id, 'name': name, 'email': email, 'phone': phone, 'address': address}
        else:
            try:
                conn.execute(
                    'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
                    (name, email, phone, address, user_id)
                )
                conn.commit()
                user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
                message = 'Profile updated successfully!'
            except Exception as e:
                error = f'Error updating profile: {str(e)}'
    
    conn.close()
    return render_template_string(EDIT_TEMPLATE, base=BASE_TEMPLATE, title=f'Edit Profile - ID {user_id}',
                                  user=user, message=message, error=error)

@app.route('/delete/<int:user_id>')
def delete_profile(user_id):
    conn = get_db()
    conn.execute