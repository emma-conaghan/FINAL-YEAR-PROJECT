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
        .navbar { background: #2c3e50; padding: 15px 30px; display: flex; align-items: center; gap: 20px; }
        .navbar a { color: white; text-decoration: none; font-size: 16px; padding: 8px 16px; border-radius: 4px; }
        .navbar a:hover { background: #34495e; }
        .navbar .brand { font-size: 20px; font-weight: bold; margin-right: 20px; }
        .container { max-width: 700px; margin: 40px auto; padding: 0 20px; }
        .card { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1 { margin-bottom: 20px; color: #2c3e50; }
        h2 { margin-bottom: 15px; color: #2c3e50; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="text"], input[type="email"], input[type="tel"], textarea {
            width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd;
            border-radius: 4px; font-size: 14px;
        }
        textarea { resize: vertical; min-height: 80px; }
        button, .btn { background: #2c3e50; color: white; padding: 10px 24px; border: none;
            border-radius: 4px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #34495e; }
        .btn-edit { background: #e67e22; }
        .btn-edit:hover { background: #d35400; }
        .btn-delete { background: #e74c3c; }
        .btn-delete:hover { background: #c0392b; }
        .success { background: #d4edda; color: #155724; padding: 12px; border-radius: 4px; margin-bottom: 20px; }
        .error { background: #f8d7da; color: #721c24; padding: 12px; border-radius: 4px; margin-bottom: 20px; }
        .profile-field { margin-bottom: 12px; }
        .profile-field .field-label { font-weight: bold; color: #777; font-size: 13px; text-transform: uppercase; }
        .profile-field .field-value { font-size: 16px; margin-top: 3px; }
        .user-list { list-style: none; }
        .user-list li { padding: 12px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .user-list li:last-child { border-bottom: none; }
        .search-form { display: flex; gap: 10px; margin-bottom: 20px; }
        .search-form input { flex: 1; margin-bottom: 0; }
        .actions { display: flex; gap: 10px; margin-top: 20px; }
        .id-badge { background: #2c3e50; color: white; padding: 2px 10px; border-radius: 12px; font-size: 13px; margin-right: 10px; }
    </style>
</head>
<body>
    <div class="navbar">
        <a href="/" class="brand">UserProfiles</a>
        <a href="/">Home</a>
        <a href="/create">Create Account</a>
        <a href="/lookup">View Profile</a>
        <a href="/users">All Users</a>
    </div>
    <div class="container">
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

HOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>Welcome to UserProfiles</h1>
    <p style="margin-bottom:20px;">A simple profile management application. Create an account, update your information, or look up profiles by ID.</p>
    <div style="display:flex; gap:10px; flex-wrap:wrap;">
        <a href="/create" class="btn">Create Account</a>
        <a href="/lookup" class="btn" style="background:#27ae60;">View Profile</a>
        <a href="/users" class="btn" style="background:#8e44ad;">All Users</a>
    </div>
</div>
{% endblock %}
'''

CREATE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>Create Account</h1>
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    <form method="POST" action="/create">
        <label for="name">Full Name *</label>
        <input type="text" id="name" name="name" required value="{{ values.get('name', '') }}">

        <label for="email">Email Address *</label>
        <input type="email" id="email" name="email" required value="{{ values.get('email', '') }}">

        <label for="phone">Phone Number</label>
        <input type="tel" id="phone" name="phone" value="{{ values.get('phone', '') }}">

        <label for="address">Address</label>
        <textarea id="address" name="address">{{ values.get('address', '') }}</textarea>

        <button type="submit">Create Account</button>
    </form>
</div>
{% endblock %}
'''

PROFILE_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    {% if success %}
    <div class="success">{{ success }}</div>
    {% endif %}
    <h1>Profile Details</h1>
    <div class="profile-field">
        <div class="field-label">Account ID</div>
        <div class="field-value"><span class="id-badge">#{{ user['id'] }}</span></div>
    </div>
    <div class="profile-field">
        <div class="field-label">Full Name</div>
        <div class="field-value">{{ user['name'] }}</div>
    </div>
    <div class="profile-field">
        <div class="field-label">Email Address</div>
        <div class="field-value">{{ user['email'] }}</div>
    </div>
    <div class="profile-field">
        <div class="field-label">Phone Number</div>
        <div class="field-value">{{ user['phone'] if user['phone'] else 'Not provided' }}</div>
    </div>
    <div class="profile-field">
        <div class="field-label">Address</div>
        <div class="field-value">{{ user['address'] if user['address'] else 'Not provided' }}</div>
    </div>
    <div class="profile-field">
        <div class="field-label">Member Since</div>
        <div class="field-value">{{ user['created_at'] }}</div>
    </div>
    <div class="actions">
        <a href="/edit/{{ user['id'] }}" class="btn btn-edit">Edit Profile</a>
        <form method="POST" action="/delete/{{ user['id'] }}" style="display:inline;" onsubmit="return confirm('Are you sure you want to delete this account?');">
            <button type="submit" class="btn-delete btn">Delete Account</button>
        </form>
    </div>
</div>
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>Edit Profile <span class="id-badge">#{{ user['id'] }}</span></h1>
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    <form method="POST" action="/edit/{{ user['id'] }}">
        <label for="name">Full Name *</label>
        <input type="text" id="name" name="name" required value="{{ values.get('name', user['name']) }}">

        <label for="email">Email Address *</label>
        <input type="email" id="email" name="email" required value="{{ values.get('email', user['email']) }}">

        <label for="phone">Phone Number</label>
        <input type="tel" id="phone" name="phone" value="{{ values.get('phone', user['phone'] or '') }}">

        <label for="address">Address</label>
        <textarea id="address" name="address">{{ values.get('address', user['address'] or '') }}</textarea>

        <div style="display:flex; gap:10px;">
            <button type="submit">Save Changes</button>
            <a href="/profile/{{ user['id'] }}" class="btn" style="background:#95a5a6;">Cancel</a>
        </div>
    </form>
</div>
{% endblock %}
'''

LOOKUP_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>View Profile by ID</h1>
    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}
    <form method="GET" action="/lookup">
        <div class="search-form">
            <input type="text" name="account_id" placeholder="Enter Account ID (e.g., 1)" value="{{ account_id or '' }}">
            <button type="submit">Look Up</button>
        </div>
    </form>
</div>
{% endblock %}
'''

USERS_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<div class="card">
    <h1>All Users</h1>
    {% if users %}
    <ul class="user-list">
        {% for user in users %}
        <li>
            <div>
                <span class="id-badge">#{{ user['id'] }}</span>
                <strong>{{ user['name'] }}</strong> — {{ user['email'] }}
            </div>
            <a href="/profile/{{ user['id'] }}" class="btn" style="padding:5px 12px; font-size:13px;">View</a>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <p>No users found. <a href="/create">Create an account</a> to get started.</p>
    {% endif %}
</div>
{% endblock %}
'''


def render(template_str, **kwargs):
    from jinja2 import Environment, BaseLoader, DictLoader
    env = Environment(loader=DictLoader({'base': BASE_TEMPLATE, 'page': template_str}))
    tmpl = env.get_template('page')
    return tmpl.render(**kwargs)


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
            return render(CREATE_TEMPLATE, title='Create Account', error='Name and email are required.', values=request.form)

        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
            (name, email, phone, address)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return redirect(url_for('profile', user_id=user_id, success='Account created successfully!'))

    return render(CREATE_TEMPLATE, title='Create Account', error=None, values={})


@app.route('/profile/<int:user_id>')
def profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        return render(LOOKUP_TEMPLATE, title='View Profile', error=f'No account found with ID #{user_id}.', account_id=str(user_id))

    success = request.args.get('success')
    return render(PROFILE_TEMPLATE, title=f'Profile - {user["name"]}', user=user, success=success)


@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    if not user:
        conn.close()
        return render(LOOKUP_TEMPLATE, title='View Profile', error=f'No account found with ID #{user_id}.', account_id=str(user_id))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            conn.close()
            return render(EDIT_TEMPLATE, title='Edit Profile', user=user, error='Name and email are required.', values=request.form)

        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        conn.close()
        return redirect(url_for('profile', user_id=user_id, success='Profile updated successfully!'))

    conn.close()
    return render(EDIT_TEMPLATE, title='Edit Profile', user=user, error=None, values={})


@app.route('/delete/<int:user_id>', methods=['POST'])
def delete(user_id):
    conn = get_db()
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('users'))


@app.route('/lookup')
def lookup():
    account_id = request.args.get('account_id', '').strip()
    if account_id:
        try:
            aid = int(account_id)
            return redirect(url_for('profile', user_id