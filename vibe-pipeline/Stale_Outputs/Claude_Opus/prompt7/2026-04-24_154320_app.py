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
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
        h1 { color: #333; }
        nav { background: #333; padding: 10px 20px; border-radius: 5px; margin-bottom: 30px; }
        nav a { color: white; text-decoration: none; margin-right: 20px; font-size: 14px; }
        nav a:hover { text-decoration: underline; }
        .card { background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        label { display: block; margin-top: 15px; font-weight: bold; color: #555; }
        input, textarea { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        textarea { height: 80px; resize: vertical; }
        button, .btn { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; text-decoration: none; display: inline-block; margin-top: 15px; }
        button:hover, .btn:hover { background: #45a049; }
        .btn-blue { background: #2196F3; }
        .btn-blue:hover { background: #1976D2; }
        .btn-red { background: #f44336; }
        .btn-red:hover { background: #d32f2f; }
        .detail-row { padding: 10px 0; border-bottom: 1px solid #eee; }
        .detail-label { font-weight: bold; color: #555; display: inline-block; width: 120px; }
        .detail-value { color: #333; }
        .message { padding: 12px; border-radius: 4px; margin-bottom: 15px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .search-form { display: flex; gap: 10px; align-items: end; }
        .search-form input { flex: 1; }
        .search-form button { margin-top: 0; }
        table { width: 100%; border-collapse: collapse; }
        table th, table td { padding: 10px; text-align: left; border-bottom: 1px solid #eee; }
        table th { background: #f9f9f9; font-weight: bold; color: #555; }
        table tr:hover { background: #f5f5f5; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/create">Create Account</a>
        <a href="/view">View Profile</a>
        <a href="/accounts">All Accounts</a>
    </nav>
    {% block content %}{% endblock %}
</body>
</html>
'''

HOME_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>User Profile Manager</h1>
<div class="card">
    <h2>Welcome!</h2>
    <p>Manage user accounts and profiles with this simple application.</p>
    <p>
        <a href="/create" class="btn">Create New Account</a>
        <a href="/view" class="btn btn-blue" style="margin-left: 10px;">View a Profile</a>
    </p>
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
<div class="card">
    <form method="POST" action="/create">
        <label for="name">Name *</label>
        <input type="text" id="name" name="name" required placeholder="Enter full name">

        <label for="email">Email *</label>
        <input type="email" id="email" name="email" required placeholder="Enter email address">

        <label for="phone">Phone Number</label>
        <input type="text" id="phone" name="phone" placeholder="Enter phone number">

        <label for="address">Address</label>
        <textarea id="address" name="address" placeholder="Enter address"></textarea>

        <button type="submit">Create Account</button>
    </form>
</div>
{% endblock %}
'''

VIEW_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>View Profile</h1>
<div class="card">
    <form method="GET" action="/view">
        <label for="account_id">Enter Account ID</label>
        <div class="search-form">
            <input type="number" id="account_id" name="id" placeholder="e.g. 1" value="{{ search_id or '' }}" required min="1">
            <button type="submit" class="btn-blue" style="background: #2196F3;">Search</button>
        </div>
    </form>
</div>

{% if error %}
<div class="message error">{{ error }}</div>
{% endif %}

{% if user %}
<div class="card">
    <h2>Profile Details</h2>
    <div class="detail-row">
        <span class="detail-label">Account ID:</span>
        <span class="detail-value">{{ user['id'] }}</span>
    </div>
    <div class="detail-row">
        <span class="detail-label">Name:</span>
        <span class="detail-value">{{ user['name'] }}</span>
    </div>
    <div class="detail-row">
        <span class="detail-label">Email:</span>
        <span class="detail-value">{{ user['email'] }}</span>
    </div>
    <div class="detail-row">
        <span class="detail-label">Phone:</span>
        <span class="detail-value">{{ user['phone'] or 'Not provided' }}</span>
    </div>
    <div class="detail-row">
        <span class="detail-label">Address:</span>
        <span class="detail-value">{{ user['address'] or 'Not provided' }}</span>
    </div>
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
<div class="card">
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
        <a href="/view?id={{ user['id'] }}" class="btn btn-blue" style="margin-left: 10px;">Cancel</a>
    </form>
</div>
{% endblock %}
'''

ACCOUNTS_TEMPLATE = '''
{% extends "base" %}
{% block content %}
<h1>All Accounts</h1>
<div class="card">
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
    <p>No accounts found. <a href="/create">Create one now!</a></p>
    {% endif %}
</div>
{% endblock %}
'''


class TemplateLoader:
    templates = {
        'base': BASE_TEMPLATE,
        'home': HOME_TEMPLATE,
        'create': CREATE_TEMPLATE,
        'view': VIEW_TEMPLATE,
        'edit': EDIT_TEMPLATE,
        'accounts': ACCOUNTS_TEMPLATE,
    }


from jinja2 import BaseLoader, TemplateNotFound

class DictLoader(BaseLoader):
    def __init__(self, templates):
        self.templates = templates

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, template, lambda: True
        raise TemplateNotFound(template)


app.jinja_loader = DictLoader(TemplateLoader.templates)


@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)


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

    return render_template_string(CREATE_TEMPLATE, message=message, message_type=message_type)


@app.route('/view')
def view():
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
            error = 'Please enter a valid numeric Account ID.'

    return render_template_string(VIEW_TEMPLATE, user=user, error=error, search_id=search_id)


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

    return render_template_string(EDIT_TEMPLATE, user=user, message=message, message_type=message_type)


@app.route('/accounts')
def accounts():
    conn = get_db()
    users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    return render_template_string(ACCOUNTS_TEMPLATE, users=users)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)