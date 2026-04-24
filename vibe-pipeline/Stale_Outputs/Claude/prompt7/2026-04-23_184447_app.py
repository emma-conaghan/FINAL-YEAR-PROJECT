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
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT,
                address TEXT,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>User Profile App</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
        h1, h2 { color: #333; }
        form { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #45a049; }
        .nav { margin-bottom: 20px; }
        .nav a { margin-right: 15px; color: #4CAF50; text-decoration: none; font-weight: bold; }
        .nav a:hover { text-decoration: underline; }
        .flash { background: #dff0d8; border: 1px solid #3c763d; padding: 10px; margin-bottom: 15px; border-radius: 4px; color: #3c763d; }
        .flash.error { background: #f2dede; border-color: #a94442; color: #a94442; }
        .profile-box { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .profile-box p { margin: 8px 0; }
        .label { font-weight: bold; color: #555; }
    </style>
</head>
<body>
    <div class="nav">
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/view">View Profile</a>
    </div>
    {% for message in get_flashed_messages() %}
        <div class="flash">{{ message }}</div>
    {% endfor %}
    {% block content %}{% endblock %}
</body>
</html>
'''

HOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h1>Welcome to the User Profile App</h1>
<p>Use the navigation above to:</p>
<ul>
    <li><a href="/register">Create a new account</a></li>
    <li><a href="/view">View a profile by Account ID</a></li>
</ul>
{% endblock %}
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>Create Account</h2>
<form method="POST">
    <label>Name:</label>
    <input type="text" name="name" required value="{{ form.name or '' }}">
    <label>Email:</label>
    <input type="email" name="email" required value="{{ form.email or '' }}">
    <label>Password:</label>
    <input type="password" name="password" required>
    <label>Phone:</label>
    <input type="text" name="phone" value="{{ form.phone or '' }}">
    <label>Address:</label>
    <textarea name="address" rows="3">{{ form.address or '' }}</textarea>
    <button type="submit">Create Account</button>
</form>
{% endblock %}
''')

UPDATE_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>Update Profile (Account ID: {{ user.id }})</h2>
<form method="POST">
    <label>Name:</label>
    <input type="text" name="name" required value="{{ user.name }}">
    <label>Email:</label>
    <input type="email" name="email" required value="{{ user.email }}">
    <label>Phone:</label>
    <input type="text" name="phone" value="{{ user.phone or '' }}">
    <label>Address:</label>
    <textarea name="address" rows="3">{{ user.address or '' }}</textarea>
    <button type="submit">Update Profile</button>
</form>
{% endblock %}
''')

VIEW_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>View Profile by Account ID</h2>
<form method="GET" action="/view">
    <label>Account ID:</label>
    <input type="number" name="id" required value="{{ search_id or '' }}">
    <button type="submit">Search</button>
</form>
{% if user %}
<br>
<div class="profile-box">
    <h3>Profile Details</h3>
    <p><span class="label">Account ID:</span> {{ user.id }}</p>
    <p><span class="label">Name:</span> {{ user.name }}</p>
    <p><span class="label">Email:</span> {{ user.email }}</p>
    <p><span class="label">Phone:</span> {{ user.phone or 'N/A' }}</p>
    <p><span class="label">Address:</span> {{ user.address or 'N/A' }}</p>
    <br>
    <a href="/update/{{ user.id }}"><button type="button">Edit Profile</button></a>
</div>
{% elif searched %}
<p>No user found with that Account ID.</p>
{% endif %}
{% endblock %}
''')

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = {}
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        form = {'name': name, 'email': email, 'phone': phone, 'address': address}
        if not name or not email or not password:
            flash('Name, email, and password are required.')
            return render_template_string(REGISTER_TEMPLATE, form=form)
        try:
            with get_db() as conn:
                cursor = conn.execute(
                    'INSERT INTO users (name, email, phone, address, password) VALUES (?, ?, ?, ?, ?)',
                    (name, email, phone, address, password)
                )
                conn.commit()
                user_id = cursor.lastrowid
            flash(f'Account created successfully! Your Account ID is: {user_id}')
            return redirect(url_for('view_profile') + f'?id={user_id}')
        except sqlite3.IntegrityError:
            flash('An account with that email already exists.')
            return render_template_string(REGISTER_TEMPLATE, form=form)
    return render_template_string(REGISTER_TEMPLATE, form=form)

@app.route('/view', methods=['GET'])
def view_profile():
    user = None
    searched = False
    search_id = request.args.get('id', '')
    if search_id:
        searched = True
        try:
            uid = int(search_id)
            with get_db() as conn:
                user = conn.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
        except (ValueError, Exception):
            pass
    return render_template_string(VIEW_TEMPLATE, user=user, searched=searched, search_id=search_id)

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update_profile(user_id):
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        flash('User not found.')
        return redirect(url_for('view_profile'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            flash('Name and email are required.')
            return render_template_string(UPDATE_TEMPLATE, user=user)
        try:
            with get_db() as conn:
                conn.execute(
                    'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                    (name, email, phone, address, user_id)
                )
                conn.commit()
            flash('Profile updated successfully!')
            return redirect(url_for('view_profile') + f'?id={user_id}')
        except sqlite3.IntegrityError:
            flash('That email is already in use by another account.')
            return render_template_string(UPDATE_TEMPLATE, user=user)
    return render_template_string(UPDATE_TEMPLATE, user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)