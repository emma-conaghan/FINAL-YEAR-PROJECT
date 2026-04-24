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
        button, .btn { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; text-decoration: none; display: inline-block; margin-top: 20px; }
        button:hover, .btn:hover { background: #0056b3; }
        .btn-success { background: #28a745; }
        .btn-success:hover { background: #1e7e34; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        .profile-detail { padding: 10px 0; border-bottom: 1px solid #eee; }
        .profile-detail strong { display: inline-block; width: 120px; color: #555; }
        .message { padding: 15px; margin-bottom: 20px; border-radius: 4px; }
        .message-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .message-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .search-form { display: flex; gap: 10px; align-items: end; }
        .search-form input { flex: 1; }
        .search-form button { margin-top: 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: bold; color: #555; }
        tr:hover { background: #f8f9fa; }
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
<h1>User Profile Management</h1>
<p>Welcome to the user profile management system. Use the navigation above to:</p>
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
<div class="message message-success">{{ message }} <a href="/view?id={{ new_id }}">View Profile</a></div>
{% endif %}
{% if error %}
<div class="message message-error">{{ error }}</div>
{% endif %}
<form method="POST" action="/create">
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

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>View Profile</h1>
<form method="GET" action="/view" class="search-form">
    <div style="flex:1">
        <label for="id">Enter Account ID</label>
        <input type="number" id="id" name="id" min="1" value="{{ search_id or '' }}" placeholder="e.g. 1">
    </div>
    <button type="submit">Search</button>
</form>

{% if error %}
<div class="message message-error" style="margin-top:20px">{{ error }}</div>
{% endif %}

{% if user %}
<div style="margin-top: 30px;">
    <h2>Profile Details</h2>
    <div class="profile-detail"><strong>Account ID:</strong> {{ user['id'] }}</div>
    <div class="profile-detail"><strong>Name:</strong> {{ user['name'] }}</div>
    <div class="profile-detail"><strong>Email:</strong> {{ user['email'] }}</div>
    <div class="profile-detail"><strong>Phone:</strong> {{ user['phone'] or 'Not provided' }}</div>
    <div class="profile-detail"><strong>Address:</strong> {{ user['address'] or 'Not provided' }}</div>
    <a href="/edit/{{ user['id'] }}" class="btn btn-success" style="margin-right:10px">Edit Profile</a>
    <a href="/delete/{{ user['id'] }}" class="btn btn-danger" onclick="return confirm('Are you sure you want to delete this account?')">Delete Account</a>
</div>
{% endif %}
{% endblock %}
'''

EDIT_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>Edit Profile - Account #{{ user['id'] }}</h1>
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
    <input type="tel" id="phone" name="phone" value="{{ user['phone'] or '' }}">

    <label for="address">Address</label>
    <textarea id="address" name="address">{{ user['address'] or '' }}</textarea>

    <button type="submit">Update Profile</button>
    <a href="/view?id={{ user['id'] }}" class="btn" style="background:#6c757d; margin-left:10px;">Cancel</a>
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
            <td>{{ user['phone'] or '-' }}</td>
            <td>
                <a href="/view?id={{ user['id'] }}">View</a> |
                <a href="/edit/{{ user['id'] }}">Edit</a>
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p>No accounts found. <a href="/create">Create one</a>.</p>
{% endif %}
{% endblock %}
'''

def render(template_str, **kwargs):
    from jinja2 import Environment, BaseLoader
    env = Environment(loader=BaseLoader())
    env.globals['url_for'] = url_for
    base_tmpl = env.from_string(BASE_TEMPLATE)
    env.globals['base'] = base_tmpl

    class DictLoader:
        pass

    full_template = BASE_TEMPLATE.replace('{% block content %}{% endblock %}',
                                           template_str.replace('{% extends "base" %}', '').replace('{% block content %}', '').replace('{% endblock %}', ''))
    return render_template_string(full_template, title=kwargs.get('title', 'User Profiles'), **kwargs)


@app.route('/')
def home():
    return render(HOME_TEMPLATE, title='Home - User Profiles')


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
        cursor = conn.execute('INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
                              (name, email, phone, address))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return render(CREATE_TEMPLATE, title='Create Account', message=f'Account created successfully! Your Account ID is {new_id}.', new_id=new_id, values={})

    return render(CREATE_TEMPLATE, title='Create Account', values={})


@app.route('/view')
def view():
    search_id = request.args.get('id', '').strip()
    if not search_id:
        return render(VIEW_TEMPLATE, title='View Profile', search_id=None, user=None)

    try:
        search_id = int(search_id)
    except ValueError:
        return render(VIEW_TEMPLATE, title='View Profile', search_id=search_id, user=None, error='Please enter a valid numeric Account ID.')

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (search_id,)).fetchone()
    conn.close()

    if not user:
        return render(VIEW_TEMPLATE, title='View Profile', search_id=search_id, user=None, error=f'No account found with ID {search_id}.')

    return render(VIEW_TEMPLATE, title=f'Profile - {user["name"]}', search_id=search_id, user=dict(user))


@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    if not user:
        conn.close()
        return redirect(url_for('view'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            conn.close()
            user_dict = dict(user)
            user_dict.update({'name': name, 'email': email, 'phone': phone, 'address': address})
            return render(EDIT_TEMPLATE, title=f'Edit Profile - {name}', user=user_dict, error='Name and email are required.')

        conn.execute('UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                     (name, email, phone, address, user_id))
        conn.commit()

        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()

        return render(EDIT_TEMPLATE, title=f'Edit Profile - {name}', user=dict(user), message='Profile updated successfully!')

    conn.close()
    return render(EDIT_TEMPLATE, title=f'Edit Profile - {user["name"]}', user=dict(user))


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
    users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    users_list = [dict(u) for u in users]
    return render(ACCOUNTS_TEMPLATE, title='All Accounts', users=users_list)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)