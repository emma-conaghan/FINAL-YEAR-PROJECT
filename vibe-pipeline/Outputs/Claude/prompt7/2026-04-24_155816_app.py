from flask import Flask, request, redirect, url_for, render_template_string, session
import sqlite3
import os
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

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
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
    h1, h2 { color: #333; }
    input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button, .btn { background: #4CAF50; color: white; padding: 10px 18px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
    button:hover, .btn:hover { background: #45a049; }
    .btn-red { background: #e74c3c; }
    .btn-red:hover { background: #c0392b; }
    .btn-blue { background: #3498db; }
    .btn-blue:hover { background: #2980b9; }
    .card { background: white; padding: 20px; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .error { color: red; margin-bottom: 10px; }
    .success { color: green; margin-bottom: 10px; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 12px; color: #3498db; text-decoration: none; }
    nav a:hover { text-decoration: underline; }
    label { font-weight: bold; color: #555; }
    .profile-field { margin-bottom: 10px; }
    .profile-label { font-weight: bold; color: #555; display: inline-block; width: 130px; }
</style>
"""

REGISTER_PAGE = BASE_STYLE + """
<div class="card">
    <h2>Create Account</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <form method="POST" action="/register">
        <label>Username</label>
        <input type="text" name="username" required>
        <label>Password</label>
        <input type="password" name="password" required>
        <label>Full Name</label>
        <input type="text" name="name">
        <label>Email</label>
        <input type="email" name="email">
        <label>Phone Number</label>
        <input type="text" name="phone">
        <label>Address</label>
        <textarea name="address" rows="3"></textarea>
        <button type="submit">Register</button>
    </form>
    <p>Already have an account? <a href="/login">Login here</a></p>
</div>
"""

LOGIN_PAGE = BASE_STYLE + """
<div class="card">
    <h2>Login</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <form method="POST" action="/login">
        <label>Username</label>
        <input type="text" name="username" required>
        <label>Password</label>
        <input type="password" name="password" required>
        <button type="submit">Login</button>
    </form>
    <p>Don't have an account? <a href="/register">Register here</a></p>
</div>
"""

DASHBOARD_PAGE = BASE_STYLE + """
<nav>
    <a href="/dashboard">Dashboard</a>
    <a href="/profile/edit">Edit Profile</a>
    <a href="/view">View Profile by ID</a>
    <a href="/logout" class="btn btn-red" style="padding:4px 10px; font-size:0.9em;">Logout</a>
</nav>
<div class="card">
    <h2>Welcome, {{ user.username }}!</h2>
    <p>Account ID: <strong>{{ user.id }}</strong></p>
    <div class="profile-field"><span class="profile-label">Full Name:</span> {{ user.name or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Email:</span> {{ user.email or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Phone:</span> {{ user.phone or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Address:</span> {{ user.address or 'Not set' }}</div>
    <br>
    <a href="/profile/edit" class="btn">Edit Profile</a>
    <a href="/view" class="btn btn-blue" style="margin-left:10px;">View Profile by ID</a>
</div>
"""

EDIT_PROFILE_PAGE = BASE_STYLE + """
<nav>
    <a href="/dashboard">Dashboard</a>
    <a href="/profile/edit">Edit Profile</a>
    <a href="/view">View Profile by ID</a>
    <a href="/logout" class="btn btn-red" style="padding:4px 10px; font-size:0.9em;">Logout</a>
</nav>
<div class="card">
    <h2>Edit Profile</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    {% if success %}<p class="success">{{ success }}</p>{% endif %}
    <form method="POST" action="/profile/edit">
        <label>Full Name</label>
        <input type="text" name="name" value="{{ user.name or '' }}">
        <label>Email</label>
        <input type="email" name="email" value="{{ user.email or '' }}">
        <label>Phone Number</label>
        <input type="text" name="phone" value="{{ user.phone or '' }}">
        <label>Address</label>
        <textarea name="address" rows="3">{{ user.address or '' }}</textarea>
        <button type="submit">Save Changes</button>
        <a href="/dashboard" class="btn btn-red" style="margin-left:10px;">Cancel</a>
    </form>
</div>
"""

VIEW_PROFILE_PAGE = BASE_STYLE + """
<nav>
    <a href="/dashboard">Dashboard</a>
    <a href="/profile/edit">Edit Profile</a>
    <a href="/view">View Profile by ID</a>
    <a href="/logout" class="btn btn-red" style="padding:4px 10px; font-size:0.9em;">Logout</a>
</nav>
<div class="card">
    <h2>View Profile by Account ID</h2>
    <form method="GET" action="/view">
        <label>Account ID</label>
        <input type="number" name="id" value="{{ search_id or '' }}" required>
        <button type="submit">Search</button>
    </form>
</div>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if profile %}
<div class="card">
    <h2>Profile Details</h2>
    <div class="profile-field"><span class="profile-label">Account ID:</span> {{ profile.id }}</div>
    <div class="profile-field"><span class="profile-label">Username:</span> {{ profile.username }}</div>
    <div class="profile-field"><span class="profile-label">Full Name:</span> {{ profile.name or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Email:</span> {{ profile.email or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Phone:</span> {{ profile.phone or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Address:</span> {{ profile.address or 'Not set' }}</div>
</div>
{% endif %}
"""

VIEW_PROFILE_PUBLIC_PAGE = BASE_STYLE + """
<nav>
    <a href="/register">Register</a>
    <a href="/login">Login</a>
</nav>
<div class="card">
    <h2>View Profile by Account ID</h2>
    <form method="GET" action="/view">
        <label>Account ID</label>
        <input type="number" name="id" value="{{ search_id or '' }}" required>
        <button type="submit">Search</button>
    </form>
</div>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
{% if profile %}
<div class="card">
    <h2>Profile Details</h2>
    <div class="profile-field"><span class="profile-label">Account ID:</span> {{ profile.id }}</div>
    <div class="profile-field"><span class="profile-label">Username:</span> {{ profile.username }}</div>
    <div class="profile-field"><span class="profile-label">Full Name:</span> {{ profile.name or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Email:</span> {{ profile.email or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Phone:</span> {{ profile.phone or 'Not set' }}</div>
    <div class="profile-field"><span class="profile-label">Address:</span> {{ profile.address or 'Not set' }}</div>
</div>
{% endif %}
"""

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not username or not password:
            error = 'Username and password are required.'
        else:
            conn = get_db()
            try:
                conn.execute(
                    'INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                    (username, hash_password(password), name, email, phone, address)
                )
                conn.commit()
                conn.close()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
                conn.close()

    return render_template_string(REGISTER_PAGE, error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ? AND password = ?',
            (username, hash_password(password))
        ).fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password.'
    return render_template_string(LOGIN_PAGE, error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    if not user:
        session.clear()
        return redirect(url_for('login'))
    return render_template_string(DASHBOARD_PAGE, user=user)

@app.route('/profile/edit', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    if not user:
        session.clear()
        return redirect(url_for('login'))

    error = None
    success = None

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        conn = get_db()
        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, session['user_id'])
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        success = 'Profile updated successfully!'

    return render_template_string(EDIT_PROFILE_PAGE, user=user, error=error, success=success)

@app.route('/view', methods=['GET'])
def view_profile():
    profile = None
    error = None
    search_id = request.args.get('id', '').strip()

    if search_id:
        try:
            uid = int(search_id)
            conn = get_db()
            profile = conn.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
            conn.close()
            if not profile:
                error = f'No account found with ID {uid}.'
        except ValueError:
            error = 'Please enter a valid numeric ID.'

    if 'user_id' in session:
        return render_template_string(VIEW_PROFILE_PAGE, profile=profile, error=error, search_id=search_id)
    else:
        return render_template_string(VIEW_PROFILE_PUBLIC_PAGE, profile=profile, error=error, search_id=search_id)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)