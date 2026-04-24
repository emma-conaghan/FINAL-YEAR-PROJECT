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
            name TEXT,
            email TEXT UNIQUE,
            phone TEXT,
            address TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
    h1 { color: #333; }
    input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button, .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; margin-top: 6px; }
    button:hover, .btn:hover { background: #45a049; }
    .btn-secondary { background: #2196F3; }
    .btn-secondary:hover { background: #1976D2; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .flash { padding: 10px; background: #f8d7da; color: #721c24; border-radius: 4px; margin-bottom: 16px; }
    .flash-success { background: #d4edda; color: #155724; }
    nav a { margin-right: 16px; color: #2196F3; text-decoration: none; }
    label { font-weight: bold; }
    .profile-field { margin-bottom: 10px; }
    .profile-label { font-weight: bold; color: #555; }
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
        <a href="{{ url_for('view_profile_form') }}">View Profile</a>
    </nav>
    <div class="card">
        <h2>Welcome</h2>
        <p>Use this app to create an account and manage your profile information.</p>
        <a href="{{ url_for('create_account') }}" class="btn">Create Account</a>
        <a href="{{ url_for('view_profile_form') }}" class="btn btn-secondary" style="margin-left:10px;">View Profile by ID</a>
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
        <a href="{{ url_for('view_profile_form') }}">View Profile</a>
    </nav>
    {% for msg in get_flashed_messages() %}
        <div class="flash">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <form method="POST">
            <label>Name</label>
            <input type="text" name="name" value="{{ form.get('name', '') }}" required>
            <label>Email</label>
            <input type="email" name="email" value="{{ form.get('email', '') }}" required>
            <label>Phone Number</label>
            <input type="text" name="phone" value="{{ form.get('phone', '') }}">
            <label>Address</label>
            <textarea name="address" rows="3">{{ form.get('address', '') }}</textarea>
            <label>Password</label>
            <input type="password" name="password" required>
            <button type="submit">Create Account</button>
        </form>
    </div>
</body>
</html>
'''

ACCOUNT_CREATED_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Account Created</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>Account Created!</h1>
    <nav>
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('view_profile_form') }}">View Profile</a>
    </nav>
    <div class="card">
        <p>Your account has been successfully created.</p>
        <p><strong>Your Account ID: {{ user_id }}</strong></p>
        <p>Keep this ID to view and update your profile.</p>
        <a href="{{ url_for('update_profile', user_id=user_id) }}" class="btn">Update Profile</a>
        <a href="{{ url_for('view_profile', user_id=user_id) }}" class="btn btn-secondary" style="margin-left:10px;">View Profile</a>
    </div>
</body>
</html>
'''

VIEW_PROFILE_FORM_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>View Profile</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>View Profile</h1>
    <nav>
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('create_account') }}">Create Account</a>
    </nav>
    {% for msg in get_flashed_messages() %}
        <div class="flash">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <form method="POST">
            <label>Enter Account ID</label>
            <input type="number" name="user_id" min="1" required>
            <button type="submit">View Profile</button>
        </form>
    </div>
</body>
</html>
'''

VIEW_PROFILE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Profile - {{ user['name'] }}</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>User Profile</h1>
    <nav>
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('create_account') }}">Create Account</a>
        <a href="{{ url_for('view_profile_form') }}">View Another Profile</a>
    </nav>
    <div class="card">
        <h2>Profile Details</h2>
        <div class="profile-field"><span class="profile-label">Account ID:</span> {{ user['id'] }}</div>
        <div class="profile-field"><span class="profile-label">Name:</span> {{ user['name'] }}</div>
        <div class="profile-field"><span class="profile-label">Email:</span> {{ user['email'] }}</div>
        <div class="profile-field"><span class="profile-label">Phone:</span> {{ user['phone'] or 'Not provided' }}</div>
        <div class="profile-field"><span class="profile-label">Address:</span> {{ user['address'] or 'Not provided' }}</div>
        <a href="{{ url_for('update_profile', user_id=user['id']) }}" class="btn" style="margin-top:16px;">Update Profile</a>
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
        <a href="{{ url_for('view_profile_form') }}">View Profile</a>
    </nav>
    {% for msg in get_flashed_messages() %}
        <div class="flash {{ 'flash-success' if 'successfully' in msg else '' }}">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <p><strong>Account ID: {{ user['id'] }}</strong></p>
        <form method="POST">
            <label>Password (required to update)</label>
            <input type="password" name="password" required>
            <label>Name</label>
            <input type="text" name="name" value="{{ user['name'] }}" required>
            <label>Email</label>
            <input type="email" name="email" value="{{ user['email'] }}" required>
            <label>Phone Number</label>
            <input type="text" name="phone" value="{{ user['phone'] or '' }}">
            <label>Address</label>
            <textarea name="address" rows="3">{{ user['address'] or '' }}</textarea>
            <button type="submit">Update Profile</button>
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
    form_data = {}
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        password = request.form.get('password', '')

        form_data = {'name': name, 'email': email, 'phone': phone, 'address': address}

        if not name or not email or not password:
            flash('Name, email, and password are required.')
            return render_template_string(CREATE_ACCOUNT_TEMPLATE, form=form_data)

        conn = get_db()
        try:
            cursor = conn.execute(
                'INSERT INTO users (name, email, phone, address, password) VALUES (?, ?, ?, ?, ?)',
                (name, email, phone, address, password)
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return render_template_string(ACCOUNT_CREATED_TEMPLATE, user_id=user_id)
        except sqlite3.IntegrityError:
            conn.close()
            flash('An account with this email already exists.')
            return render_template_string(CREATE_ACCOUNT_TEMPLATE, form=form_data)

    return render_template_string(CREATE_ACCOUNT_TEMPLATE, form=form_data)

@app.route('/profile', methods=['GET', 'POST'])
def view_profile_form():
    if request.method == 'POST':
        user_id = request.form.get('user_id', '').strip()
        if not user_id:
            flash('Please enter an account ID.')
            return render_template_string(VIEW_PROFILE_FORM_TEMPLATE)
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(VIEW_PROFILE_FORM_TEMPLATE)

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if not user:
        flash('No account found with that ID.')
        return redirect(url_for('view_profile_form'))
    return render_template_string(VIEW_PROFILE_TEMPLATE, user=user)

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        flash('No account found with that ID.')
        return redirect(url_for('view_profile_form'))

    if request.method == 'POST':
        password = request.form.get('password', '')
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if user['password'] != password:
            flash('Incorrect password. Profile not updated.')
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

        if not name or not email:
            flash('Name and email are required.')
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

        conn = get_db()
        try:
            conn.execute(
                'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                (name, email, phone, address, user_id)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
            conn.close()
            flash('Profile updated successfully!')
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)
        except sqlite3.IntegrityError:
            conn.close()
            flash('An account with this email already exists.')
            return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

    return render_template_string(UPDATE_PROFILE_TEMPLATE, user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)