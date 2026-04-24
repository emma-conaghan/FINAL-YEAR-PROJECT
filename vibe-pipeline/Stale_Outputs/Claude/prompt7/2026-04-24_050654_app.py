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

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    form { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    input[type=text], input[type=email], input[type=password], input[type=tel], textarea {
        width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box;
        border: 1px solid #ccc; border-radius: 4px;
    }
    input[type=submit], .btn {
        background: #4CAF50; color: white; padding: 10px 20px; border: none;
        border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block;
    }
    input[type=submit]:hover, .btn:hover { background: #45a049; }
    .btn-secondary { background: #2196F3; }
    .btn-secondary:hover { background: #1976D2; }
    .flash { padding: 10px; margin: 10px 0; border-radius: 4px; background: #dff0d8; color: #3c763d; border: 1px solid #d6e9c6; }
    .flash.error { background: #f2dede; color: #a94442; border: 1px solid #ebccd1; }
    .profile-box { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .profile-box p { margin: 8px 0; }
    .profile-box strong { display: inline-block; width: 120px; color: #555; }
    .nav { margin-bottom: 20px; }
    .nav a { margin-right: 10px; color: #2196F3; text-decoration: none; }
    .nav a:hover { text-decoration: underline; }
    label { font-weight: bold; color: #444; }
</style>
"""

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>User Profiles</title>""" + BASE_STYLE + """</head>
<body>
    <h1>User Profile App</h1>
    <div class="nav">
        <a href="{{ url_for('create_account') }}">Create Account</a>
        <a href="{{ url_for('view_profile_search') }}">View Profile by ID</a>
    </div>
    {% for msg in get_flashed_messages() %}
    <div class="flash">{{ msg }}</div>
    {% endfor %}
    <p>Welcome! Use the links above to create an account or view a profile.</p>
</body>
</html>
"""

CREATE_ACCOUNT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Create Account</title>""" + BASE_STYLE + """</head>
<body>
    <h1>Create Account</h1>
    <div class="nav">
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('view_profile_search') }}">View Profile by ID</a>
    </div>
    {% for msg in get_flashed_messages() %}
    <div class="flash">{{ msg }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" required>
        <label>Email:</label>
        <input type="email" name="email" required>
        <label>Phone:</label>
        <input type="tel" name="phone">
        <label>Address:</label>
        <textarea name="address" rows="3"></textarea>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Create Account">
    </form>
</body>
</html>
"""

UPDATE_PROFILE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Update Profile</title>""" + BASE_STYLE + """</head>
<body>
    <h1>Update Profile</h1>
    <div class="nav">
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('view_profile', user_id=user['id']) }}">View Profile</a>
    </div>
    {% for msg in get_flashed_messages() %}
    <div class="flash">{{ msg }}</div>
    {% endfor %}
    <p>Account ID: <strong>{{ user['id'] }}</strong></p>
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" value="{{ user['name'] }}" required>
        <label>Email:</label>
        <input type="email" name="email" value="{{ user['email'] }}" required>
        <label>Phone:</label>
        <input type="tel" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address:</label>
        <textarea name="address" rows="3">{{ user['address'] or '' }}</textarea>
        <label>Current Password (required to save changes):</label>
        <input type="password" name="password" required>
        <label>New Password (leave blank to keep current):</label>
        <input type="password" name="new_password">
        <input type="submit" value="Update Profile">
    </form>
</body>
</html>
"""

VIEW_PROFILE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>View Profile</title>""" + BASE_STYLE + """</head>
<body>
    <h1>User Profile</h1>
    <div class="nav">
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('view_profile_search') }}">Search Another</a>
        <a href="{{ url_for('update_profile', user_id=user['id']) }}" class="btn btn-secondary">Edit Profile</a>
    </div>
    {% for msg in get_flashed_messages() %}
    <div class="flash">{{ msg }}</div>
    {% endfor %}
    <div class="profile-box">
        <p><strong>Account ID:</strong> {{ user['id'] }}</p>
        <p><strong>Name:</strong> {{ user['name'] }}</p>
        <p><strong>Email:</strong> {{ user['email'] }}</p>
        <p><strong>Phone:</strong> {{ user['phone'] or 'N/A' }}</p>
        <p><strong>Address:</strong> {{ user['address'] or 'N/A' }}</p>
    </div>
</body>
</html>
"""

SEARCH_PROFILE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>View Profile by ID</title>""" + BASE_STYLE + """</head>
<body>
    <h1>View Profile by Account ID</h1>
    <div class="nav">
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('create_account') }}">Create Account</a>
    </div>
    {% for msg in get_flashed_messages() %}
    <div class="flash">{{ msg }}</div>
    {% endfor %}
    <form method="POST">
        <label>Account ID:</label>
        <input type="text" name="user_id" required>
        <input type="submit" value="View Profile">
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        password = request.form.get('password', '')

        if not name or not email or not password:
            flash('Name, email, and password are required.')
            return render_template_string(CREATE_ACCOUNT_TEMPLATE)

        try:
            with get_db() as conn:
                cursor = conn.execute(
                    'INSERT INTO users (name, email, phone, address, password) VALUES (?, ?, ?, ?, ?)',
                    (name, email, phone, address, password)
                )
                conn.commit()
                user_id = cursor.lastrowid
            flash(f'Account created! Your Account ID is {user_id}. Save it to access your profile.')
            return redirect(url_for('view_profile', user_id=user_id))
        except sqlite3.IntegrityError:
            flash('Email already exists. Please use a different email.')
            return render_template_string(CREATE_ACCOUNT_TEMPLATE)

    return render_template_string(CREATE_ACCOUNT_TEMPLATE)

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        flash('User not found.')
        return redirect(url_for('view_profile_search'))
    return render_template_string(VIEW_PROFILE_TEMPLATE, user=user)

@app.route('/profile/search', methods=['GET', 'POST'])
def view_profile_search():
    if request.method == 'POST':
        user_id = request.form.get('user_id', '').strip()
        if not user_id.isdigit():
            flash('Please enter a valid numeric Account ID.')
            return render_template_string(SEARCH_PROFILE_TEMPLATE)
        return redirect(url_for('view_profile', user_id=int(user_id)))
    return render_template_string(SEARCH_PROFILE_TEMPLATE)

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update_profile(user_id):
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    if not user:
        flash('User not found.')
        return redirect(url_for('view_profile_search'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        password = request.form.get('password', '')
        new_password = request.form.get('new_password', '')

        if user['password'] != password:
            flash('Incorrect password. Changes not saved.')
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

        if not name or not email:
            flash('Name and email are required.')
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

        final_password = new_password if new_password else password

        try:
            with get_db() as conn:
                conn.execute(
                    'UPDATE users SET name=?, email=?, phone=?, address=?, password=? WHERE id=?',
                    (name, email, phone, address, final_password, user_id)
                )
                conn.commit()
            flash('Profile updated successfully!')
            return redirect(url_for('view_profile', user_id=user_id))
        except sqlite3.IntegrityError:
            flash('Email already in use by another account.')
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

    return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)