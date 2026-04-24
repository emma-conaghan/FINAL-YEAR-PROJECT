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
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button, .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
    button:hover, .btn:hover { background: #45a049; }
    .btn-secondary { background: #2196F3; }
    .btn-secondary:hover { background: #1976D2; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .flash-success { background: #dff0d8; color: #3c763d; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
    .flash-error { background: #f2dede; color: #a94442; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
    label { font-weight: bold; color: #555; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 15px; color: #2196F3; text-decoration: none; }
    nav a:hover { text-decoration: underline; }
    .profile-field { margin-bottom: 12px; }
    .profile-label { font-weight: bold; color: #555; display: inline-block; width: 120px; }
</style>
'''

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>User Profiles</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>User Profile App</h1>
    <nav>
        <a href="{{ url_for('create_account') }}">Create Account</a>
        <a href="{{ url_for('view_profile_page') }}">View Profile by ID</a>
    </nav>
    <div class="card">
        <h2>Welcome</h2>
        <p>Use the links above to create an account or view a profile by account ID.</p>
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
        <a href="{{ url_for('view_profile_page') }}">View Profile by ID</a>
    </nav>
    {% for msg in get_flashed_messages(category_filter=['error']) %}
        <div class="flash-error">{{ msg }}</div>
    {% endfor %}
    {% for msg in get_flashed_messages(category_filter=['success']) %}
        <div class="flash-success">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <form method="POST">
            <label>Username (required):</label>
            <input type="text" name="username" required placeholder="Enter a username">
            <label>Password (required):</label>
            <input type="password" name="password" required placeholder="Enter a password">
            <label>Full Name:</label>
            <input type="text" name="name" placeholder="Your full name">
            <label>Email:</label>
            <input type="email" name="email" placeholder="your@email.com">
            <label>Phone Number:</label>
            <input type="text" name="phone" placeholder="e.g. 555-1234">
            <label>Address:</label>
            <textarea name="address" rows="3" placeholder="Your address"></textarea>
            <button type="submit">Create Account</button>
        </form>
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
        <a href="{{ url_for('view_profile_page') }}">View Profile by ID</a>
    </nav>
    {% for msg in get_flashed_messages(category_filter=['error']) %}
        <div class="flash-error">{{ msg }}</div>
    {% endfor %}
    {% for msg in get_flashed_messages(category_filter=['success']) %}
        <div class="flash-success">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <p><strong>Account ID:</strong> {{ user['id'] }} &nbsp; <strong>Username:</strong> {{ user['username'] }}</p>
        <form method="POST">
            <label>Full Name:</label>
            <input type="text" name="name" value="{{ user['name'] or '' }}" placeholder="Your full name">
            <label>Email:</label>
            <input type="email" name="email" value="{{ user['email'] or '' }}" placeholder="your@email.com">
            <label>Phone Number:</label>
            <input type="text" name="phone" value="{{ user['phone'] or '' }}" placeholder="e.g. 555-1234">
            <label>Address:</label>
            <textarea name="address" rows="3" placeholder="Your address">{{ user['address'] or '' }}</textarea>
            <button type="submit">Update Profile</button>
        </form>
    </div>
    <div class="card">
        <a href="{{ url_for('view_profile', user_id=user['id']) }}" class="btn btn-secondary">View My Profile</a>
    </div>
</body>
</html>
'''

VIEW_PROFILE_PAGE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>View Profile by ID</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>View Profile by Account ID</h1>
    <nav>
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('create_account') }}">Create Account</a>
    </nav>
    {% for msg in get_flashed_messages(category_filter=['error']) %}
        <div class="flash-error">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <form method="GET" action="{{ url_for('view_profile_search') }}">
            <label>Enter Account ID:</label>
            <input type="number" name="user_id" required placeholder="e.g. 1" min="1">
            <button type="submit">View Profile</button>
        </form>
    </div>
</body>
</html>
'''

VIEW_PROFILE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Profile - {{ user['username'] }}</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>Profile Details</h1>
    <nav>
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('create_account') }}">Create Account</a>
        <a href="{{ url_for('view_profile_page') }}">View Profile by ID</a>
    </nav>
    <div class="card">
        <div class="profile-field"><span class="profile-label">Account ID:</span> {{ user['id'] }}</div>
        <div class="profile-field"><span class="profile-label">Username:</span> {{ user['username'] }}</div>
        <div class="profile-field"><span class="profile-label">Full Name:</span> {{ user['name'] or 'Not set' }}</div>
        <div class="profile-field"><span class="profile-label">Email:</span> {{ user['email'] or 'Not set' }}</div>
        <div class="profile-field"><span class="profile-label">Phone:</span> {{ user['phone'] or 'Not set' }}</div>
        <div class="profile-field"><span class="profile-label">Address:</span> {{ user['address'] or 'Not set' }}</div>
    </div>
    <div class="card">
        <p>Want to update this profile? Log in below:</p>
        <form method="POST" action="{{ url_for('login_and_edit') }}">
            <input type="hidden" name="user_id" value="{{ user['id'] }}">
            <label>Password:</label>
            <input type="password" name="password" required placeholder="Enter password">
            <button type="submit">Edit Profile</button>
        </form>
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

        try:
            conn = get_db()
            conn.execute(
                'INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                (username, password, name, email, phone, address)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            conn.close()
            flash(f'Account created successfully! Your Account ID is {user["id"]}.', 'success')
            return redirect(url_for('view_profile', user_id=user['id']))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.', 'error')
            conn.close()

    return render_template_string(CREATE_ACCOUNT_TEMPLATE)

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if not user:
        flash('No user found with that ID.', 'error')
        return redirect(url_for('view_profile_page'))
    return render_template_string(VIEW_PROFILE_TEMPLATE, user=user)

@app.route('/view', methods=['GET'])
def view_profile_page():
    return render_template_string(VIEW_PROFILE_PAGE_TEMPLATE)

@app.route('/view/search', methods=['GET'])
def view_profile_search():
    user_id = request.args.get('user_id', '')
    if not user_id:
        flash('Please enter an account ID.', 'error')
        return redirect(url_for('view_profile_page'))
    try:
        user_id = int(user_id)
    except ValueError:
        flash('Invalid account ID.', 'error')
        return redirect(url_for('view_profile_page'))
    return redirect(url_for('view_profile', user_id=user_id))

@app.route('/login_edit', methods=['POST'])
def login_and_edit():
    user_id = request.form.get('user_id', '')
    password = request.form.get('password', '')
    try:
        user_id = int(user_id)
    except ValueError:
        flash('Invalid user ID.', 'error')
        return redirect(url_for('view_profile_page'))

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('view_profile_page'))

    if user['password'] != password:
        flash('Incorrect password.', 'error')
        return redirect(url_for('view_profile', user_id=user_id))

    return redirect(url_for('update_profile', user_id=user_id))

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('view_profile_page'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        conn = get_db()
        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        flash('Profile updated successfully!', 'success')
        return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

    return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)