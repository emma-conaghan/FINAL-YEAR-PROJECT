from flask import Flask, render_template_string, request, redirect, url_for, flash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'
DB_PATH = 'users.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    input[type=text], input[type=password], input[type=email], textarea {
        width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px;
    }
    input[type=submit], .btn {
        background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block;
    }
    input[type=submit]:hover, .btn:hover { background: #45a049; }
    .btn-secondary { background: #2196F3; }
    .btn-secondary:hover { background: #1976D2; }
    .flash { padding: 10px; margin: 10px 0; border-radius: 4px; }
    .flash.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .flash.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    label { font-weight: bold; color: #555; }
    .profile-field { margin-bottom: 12px; }
    .profile-field span { color: #333; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 15px; color: #2196F3; text-decoration: none; }
    nav a:hover { text-decoration: underline; }
</style>
'''

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>User Profile App</title>''' + BASE_STYLE + '''</head>
<body>
<h1>User Profile App</h1>
<nav>
    <a href="{{ url_for('create_account') }}">Create Account</a>
    <a href="{{ url_for('view_profile_search') }}">View Profile</a>
    <a href="{{ url_for('update_profile') }}">Update Profile</a>
</nav>
<div class="card">
    <p>Welcome! Use the links above to create an account, view profile details, or update your profile information.</p>
</div>
</body>
</html>
'''

CREATE_ACCOUNT_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Create Account</title>''' + BASE_STYLE + '''</head>
<body>
<h1>Create Account</h1>
<nav>
    <a href="{{ url_for('home') }}">Home</a>
    <a href="{{ url_for('view_profile_search') }}">View Profile</a>
    <a href="{{ url_for('update_profile') }}">Update Profile</a>
</nav>
{% for msg in get_flashed_messages(with_categories=true) %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
{% endfor %}
<div class="card">
    <form method="POST">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <label>Name:</label>
        <input type="text" name="name">
        <label>Email:</label>
        <input type="text" name="email">
        <label>Phone:</label>
        <input type="text" name="phone">
        <label>Address:</label>
        <input type="text" name="address">
        <input type="submit" value="Create Account">
    </form>
</div>
</body>
</html>
'''

VIEW_SEARCH_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>View Profile</title>''' + BASE_STYLE + '''</head>
<body>
<h1>View Profile by Account ID</h1>
<nav>
    <a href="{{ url_for('home') }}">Home</a>
    <a href="{{ url_for('create_account') }}">Create Account</a>
    <a href="{{ url_for('update_profile') }}">Update Profile</a>
</nav>
{% for msg in get_flashed_messages(with_categories=true) %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
{% endfor %}
<div class="card">
    <form method="POST">
        <label>Account ID:</label>
        <input type="text" name="account_id" required>
        <input type="submit" value="View Profile">
    </form>
</div>
</body>
</html>
'''

PROFILE_DETAIL_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Profile Details</title>''' + BASE_STYLE + '''</head>
<body>
<h1>Profile Details</h1>
<nav>
    <a href="{{ url_for('home') }}">Home</a>
    <a href="{{ url_for('create_account') }}">Create Account</a>
    <a href="{{ url_for('view_profile_search') }}">View Another Profile</a>
    <a href="{{ url_for('update_profile') }}">Update Profile</a>
</nav>
<div class="card">
    <div class="profile-field"><label>Account ID:</label><br><span>{{ user['id'] }}</span></div>
    <div class="profile-field"><label>Username:</label><br><span>{{ user['username'] }}</span></div>
    <div class="profile-field"><label>Name:</label><br><span>{{ user['name'] or 'Not provided' }}</span></div>
    <div class="profile-field"><label>Email:</label><br><span>{{ user['email'] or 'Not provided' }}</span></div>
    <div class="profile-field"><label>Phone:</label><br><span>{{ user['phone'] or 'Not provided' }}</span></div>
    <div class="profile-field"><label>Address:</label><br><span>{{ user['address'] or 'Not provided' }}</span></div>
</div>
</body>
</html>
'''

UPDATE_PROFILE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Update Profile</title>''' + BASE_STYLE + '''</head>
<body>
<h1>Update Profile</h1>
<nav>
    <a href="{{ url_for('home') }}">Home</a>
    <a href="{{ url_for('create_account') }}">Create Account</a>
    <a href="{{ url_for('view_profile_search') }}">View Profile</a>
</nav>
{% for msg in get_flashed_messages(with_categories=true) %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
{% endfor %}
<div class="card">
    {% if not user %}
    <h2>Login to Update Profile</h2>
    <form method="POST">
        <input type="hidden" name="step" value="login">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Login">
    </form>
    {% else %}
    <h2>Editing Profile for: {{ user['username'] }} (ID: {{ user['id'] }})</h2>
    <form method="POST">
        <input type="hidden" name="step" value="update">
        <input type="hidden" name="user_id" value="{{ user['id'] }}">
        <label>Name:</label>
        <input type="text" name="name" value="{{ user['name'] or '' }}">
        <label>Email:</label>
        <input type="text" name="email" value="{{ user['email'] or '' }}">
        <label>Phone:</label>
        <input type="text" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address:</label>
        <input type="text" name="address" value="{{ user['address'] or '' }}">
        <input type="submit" value="Update Profile">
    </form>
    {% endif %}
</div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template_string(CREATE_ACCOUNT_TEMPLATE)

        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                (username, password, name, email, phone, address)
            )
            conn.commit()
            user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            flash(f'Account created successfully! Your Account ID is: {user["id"]}', 'success')
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose a different username.', 'error')
        finally:
            conn.close()

    return render_template_string(CREATE_ACCOUNT_TEMPLATE)

@app.route('/profile', methods=['GET', 'POST'])
def view_profile_search():
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        if not account_id:
            flash('Please enter an account ID.', 'error')
            return render_template_string(VIEW_SEARCH_TEMPLATE)
        try:
            account_id_int = int(account_id)
        except ValueError:
            flash('Account ID must be a number.', 'error')
            return render_template_string(VIEW_SEARCH_TEMPLATE)

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id_int,)).fetchone()
        conn.close()

        if user is None:
            flash(f'No account found with ID: {account_id}', 'error')
            return render_template_string(VIEW_SEARCH_TEMPLATE)

        return render_template_string(PROFILE_DETAIL_TEMPLATE, user=user)

    return render_template_string(VIEW_SEARCH_TEMPLATE)

@app.route('/update', methods=['GET', 'POST'])
def update_profile():
    user = None
    if request.method == 'POST':
        step = request.form.get('step', '')

        if step == 'login':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            conn = get_db()
            user_row = conn.execute(
                'SELECT * FROM users WHERE username = ? AND password = ?', (username, password)
            ).fetchone()
            conn.close()
            if user_row is None:
                flash('Invalid username or password.', 'error')
                return render_template_string(UPDATE_PROFILE_TEMPLATE, user=None)
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user_row)

        elif step == 'update':
            user_id = request.form.get('user_id', '').strip()
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            phone = request.form.get('phone', '').strip()
            address = request.form.get('address', '').strip()

            try:
                user_id_int = int(user_id)
            except ValueError:
                flash('Invalid user ID.', 'error')
                return render_template_string(UPDATE_PROFILE_TEMPLATE, user=None)

            conn = get_db()
            conn.execute(
                'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                (name, email, phone, address, user_id_int)
            )
            conn.commit()
            user_row = conn.execute('SELECT * FROM users WHERE id = ?', (user_id_int,)).fetchone()
            conn.close()
            flash('Profile updated successfully!', 'success')
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user_row)

    return render_template_string(UPDATE_PROFILE_TEMPLATE, user=None)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)