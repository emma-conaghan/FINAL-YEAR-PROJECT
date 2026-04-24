from flask import Flask, render_template_string, request, redirect, url_for, flash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'
DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                phone TEXT,
                address TEXT
            )
        ''')
        conn.commit()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>User Profiles</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; background: #f5f5f5; color: #333; }
        h1, h2 { color: #444; }
        input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button, .btn { background: #4CAF50; color: white; padding: 10px 18px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #45a049; }
        .nav { margin-bottom: 20px; }
        .nav a { margin-right: 15px; color: #4CAF50; text-decoration: none; font-weight: bold; }
        .card { background: white; padding: 24px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .flash { padding: 10px; background: #dff0d8; border: 1px solid #3c763d; border-radius: 4px; margin-bottom: 14px; color: #3c763d; }
        .flash.error { background: #f2dede; border-color: #a94442; color: #a94442; }
        label { font-weight: bold; }
        .profile-item { margin-bottom: 10px; }
        .profile-item span { font-weight: bold; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="{{ url_for('index') }}">Home</a>
        <a href="{{ url_for('create_account') }}">Create Account</a>
        <a href="{{ url_for('view_profile_search') }}">View Profile</a>
    </div>
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% for category, message in messages %}
            <div class="flash {{ category }}">{{ message }}</div>
        {% endfor %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
'''

INDEX_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="card">
    <h1>User Profile Manager</h1>
    <p>Welcome! Use this app to create and manage user profiles.</p>
    <a href="{{ url_for('create_account') }}" class="btn">Create Account</a>
    &nbsp;
    <a href="{{ url_for('view_profile_search') }}" class="btn" style="background:#2196F3;">View Profile by ID</a>
</div>
{% endblock %}
''')

CREATE_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="card">
    <h2>Create Account</h2>
    <form method="POST">
        <label>Name</label>
        <input type="text" name="name" placeholder="Full Name" required>
        <label>Email</label>
        <input type="email" name="email" placeholder="Email Address" required>
        <label>Phone Number</label>
        <input type="text" name="phone" placeholder="Phone Number">
        <label>Address</label>
        <textarea name="address" placeholder="Address" rows="3"></textarea>
        <button type="submit">Create Account</button>
    </form>
</div>
{% endblock %}
''')

EDIT_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="card">
    <h2>Update Profile (Account ID: {{ user['id'] }})</h2>
    <form method="POST">
        <label>Name</label>
        <input type="text" name="name" value="{{ user['name'] }}" required>
        <label>Email</label>
        <input type="email" name="email" value="{{ user['email'] }}" required>
        <label>Phone Number</label>
        <input type="text" name="phone" value="{{ user['phone'] }}">
        <label>Address</label>
        <textarea name="address" rows="3">{{ user['address'] }}</textarea>
        <button type="submit">Update Profile</button>
        &nbsp;
        <a href="{{ url_for('view_profile', account_id=user['id']) }}" class="btn" style="background:#2196F3;">View Profile</a>
    </form>
</div>
{% endblock %}
''')

VIEW_PROFILE_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="card">
    <h2>Profile Details (Account ID: {{ user['id'] }})</h2>
    <div class="profile-item"><span>Name:</span> {{ user['name'] }}</div>
    <div class="profile-item"><span>Email:</span> {{ user['email'] }}</div>
    <div class="profile-item"><span>Phone:</span> {{ user['phone'] if user['phone'] else 'N/A' }}</div>
    <div class="profile-item"><span>Address:</span> {{ user['address'] if user['address'] else 'N/A' }}</div>
    <br>
    <a href="{{ url_for('edit_profile', account_id=user['id']) }}" class="btn">Edit Profile</a>
</div>
{% endblock %}
''')

SEARCH_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="card">
    <h2>View Profile by Account ID</h2>
    <form method="POST">
        <label>Account ID</label>
        <input type="number" name="account_id" placeholder="Enter Account ID" required min="1">
        <button type="submit">Search</button>
    </form>
</div>
{% endblock %}
''')

@app.route('/')
def index():
    return render_template_string(INDEX_TEMPLATE)

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            flash('Name and Email are required.', 'error')
            return render_template_string(CREATE_TEMPLATE)
        with get_db() as conn:
            cursor = conn.execute(
                'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
                (name, email, phone, address)
            )
            conn.commit()
            new_id = cursor.lastrowid
        flash(f'Account created successfully! Your Account ID is {new_id}. Please save it.', 'message')
        return redirect(url_for('view_profile', account_id=new_id))
    return render_template_string(CREATE_TEMPLATE)

@app.route('/profile/<int:account_id>')
def view_profile(account_id):
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    if not user:
        flash(f'No account found with ID {account_id}.', 'error')
        return redirect(url_for('view_profile_search'))
    return render_template_string(VIEW_PROFILE_TEMPLATE, user=user)

@app.route('/profile/<int:account_id>/edit', methods=['GET', 'POST'])
def edit_profile(account_id):
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
    if not user:
        flash(f'No account found with ID {account_id}.', 'error')
        return redirect(url_for('view_profile_search'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            flash('Name and Email are required.', 'error')
            return render_template_string(EDIT_TEMPLATE, user=user)
        with get_db() as conn:
            conn.execute(
                'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                (name, email, phone, address, account_id)
            )
            conn.commit()
        flash('Profile updated successfully!', 'message')
        return redirect(url_for('view_profile', account_id=account_id))
    return render_template_string(EDIT_TEMPLATE, user=user)

@app.route('/search', methods=['GET', 'POST'])
def view_profile_search():
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        if not account_id.isdigit():
            flash('Please enter a valid numeric Account ID.', 'error')
            return render_template_string(SEARCH_TEMPLATE)
        return redirect(url_for('view_profile', account_id=int(account_id)))
    return render_template_string(SEARCH_TEMPLATE)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)