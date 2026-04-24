from flask import Flask, Flask, request, redirect, url_for, render_template_string
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
            phone TEXT,
            address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

BASE_STYLE = """
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; background: #f4f6f9; color: #333; }
    .container { max-width: 700px; margin: 40px auto; padding: 20px; }
    h1, h2 { margin-bottom: 20px; color: #2c3e50; }
    nav { background: #2c3e50; padding: 15px 30px; }
    nav a { color: #ecf0f1; text-decoration: none; margin-right: 20px; font-size: 16px; }
    nav a:hover { text-decoration: underline; }
    .card { background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
    label { display: block; margin-bottom: 5px; font-weight: bold; margin-top: 15px; }
    input[type="text"], input[type="email"], input[type="tel"], textarea {
        width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px;
    }
    textarea { height: 80px; resize: vertical; }
    button, .btn { 
        display: inline-block; padding: 10px 25px; background: #3498db; color: white; 
        border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 20px;
        text-decoration: none;
    }
    button:hover, .btn:hover { background: #2980b9; }
    .btn-success { background: #27ae60; }
    .btn-success:hover { background: #219a52; }
    .detail-row { padding: 12px 0; border-bottom: 1px solid #eee; }
    .detail-label { font-weight: bold; color: #7f8c8d; font-size: 13px; text-transform: uppercase; }
    .detail-value { margin-top: 4px; font-size: 16px; }
    .message { padding: 12px 20px; border-radius: 4px; margin-bottom: 20px; }
    .message-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .message-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .user-list { list-style: none; }
    .user-list li { padding: 12px 15px; background: white; margin-bottom: 8px; border-radius: 4px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: flex; justify-content: space-between; align-items: center; }
    .user-list li a { color: #3498db; text-decoration: none; }
    .user-list li a:hover { text-decoration: underline; }
    .search-form { display: flex; gap: 10px; margin-bottom: 20px; }
    .search-form input { flex: 1; }
    .search-form button { margin-top: 0; }
</style>
"""

NAV = """
<nav>
    <a href="/">Home</a>
    <a href="/create">Create Account</a>
    <a href="/search">Find Profile</a>
    <a href="/users">All Users</a>
</nav>
"""

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>User Profile Manager</title>""" + BASE_STYLE + """</head>
<body>
""" + NAV + """
<div class="container">
    <h1>User Profile Manager</h1>
    <div class="card">
        <h2>Welcome!</h2>
        <p>Manage user profiles easily. You can:</p>
        <ul style="margin: 15px 0 15px 20px;">
            <li>Create a new account</li>
            <li>View profile details by account ID</li>
            <li>Update profile information</li>
            <li>Browse all registered users</li>
        </ul>
        <a href="/create" class="btn btn-success">Create New Account</a>
        <a href="/search" class="btn">Find a Profile</a>
    </div>
</div>
</body>
</html>
"""

CREATE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Create Account</title>""" + BASE_STYLE + """</head>
<body>
""" + NAV + """
<div class="container">
    <h1>Create New Account</h1>
    {% if error %}
    <div class="message message-error">{{ error }}</div>
    {% endif %}
    <div class="card">
        <form method="POST" action="/create">
            <label for="name">Full Name *</label>
            <input type="text" id="name" name="name" value="{{ name or '' }}" required>
            
            <label for="email">Email Address *</label>
            <input type="email" id="email" name="email" value="{{ email or '' }}" required>
            
            <label for="phone">Phone Number</label>
            <input type="tel" id="phone" name="phone" value="{{ phone or '' }}">
            
            <label for="address">Address</label>
            <textarea id="address" name="address">{{ address or '' }}</textarea>
            
            <button type="submit">Create Account</button>
        </form>
    </div>
</div>
</body>
</html>
"""

PROFILE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Profile - {{ user['name'] }}</title>""" + BASE_STYLE + """</head>
<body>
""" + NAV + """
<div class="container">
    {% if message %}
    <div class="message message-success">{{ message }}</div>
    {% endif %}
    <h1>Profile Details</h1>
    <div class="card">
        <div class="detail-row">
            <div class="detail-label">Account ID</div>
            <div class="detail-value">{{ user['id'] }}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Full Name</div>
            <div class="detail-value">{{ user['name'] }}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Email Address</div>
            <div class="detail-value">{{ user['email'] }}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Phone Number</div>
            <div class="detail-value">{{ user['phone'] or 'Not provided' }}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Address</div>
            <div class="detail-value">{{ user['address'] or 'Not provided' }}</div>
        </div>
        <div class="detail-row">
            <div class="detail-label">Member Since</div>
            <div class="detail-value">{{ user['created_at'] }}</div>
        </div>
        <a href="/edit/{{ user['id'] }}" class="btn">Edit Profile</a>
    </div>
</div>
</body>
</html>
"""

EDIT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Edit Profile</title>""" + BASE_STYLE + """</head>
<body>
""" + NAV + """
<div class="container">
    <h1>Edit Profile (ID: {{ user['id'] }})</h1>
    {% if error %}
    <div class="message message-error">{{ error }}</div>
    {% endif %}
    <div class="card">
        <form method="POST" action="/edit/{{ user['id'] }}">
            <label for="name">Full Name *</label>
            <input type="text" id="name" name="name" value="{{ user['name'] }}" required>
            
            <label for="email">Email Address *</label>
            <input type="email" id="email" name="email" value="{{ user['email'] }}" required>
            
            <label for="phone">Phone Number</label>
            <input type="tel" id="phone" name="phone" value="{{ user['phone'] or '' }}">
            
            <label for="address">Address</label>
            <textarea id="address" name="address">{{ user['address'] or '' }}</textarea>
            
            <button type="submit" class="btn-success">Save Changes</button>
            <a href="/profile/{{ user['id'] }}" class="btn" style="background:#95a5a6;">Cancel</a>
        </form>
    </div>
</div>
</body>
</html>
"""

SEARCH_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Find Profile</title>""" + BASE_STYLE + """</head>
<body>
""" + NAV + """
<div class="container">
    <h1>Find Profile by Account ID</h1>
    {% if error %}
    <div class="message message-error">{{ error }}</div>
    {% endif %}
    <div class="card">
        <form method="GET" action="/search">
            <label for="account_id">Account ID</label>
            <div class="search-form">
                <input type="text" id="account_id" name="account_id" placeholder="Enter account ID..." value="{{ account_id or '' }}">
                <button type="submit">Search</button>
            </div>
        </form>
    </div>
</div>
</body>
</html>
"""

USERS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>All Users</title>""" + BASE_STYLE + """</head>
<body>
""" + NAV + """
<div class="container">
    <h1>All Registered Users</h1>
    {% if users %}
    <ul class="user-list">
        {% for user in users %}
        <li>
            <div>
                <strong>{{ user['name'] }}</strong><br>
                <small style="color:#7f8c8d;">{{ user['email'] }} | ID: {{ user['id'] }}</small>
            </div>
            <div>
                <a href="/profile/{{ user['id'] }}">View Profile</a> | 
                <a href="/edit/{{ user['id'] }}">Edit</a>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <div class="card">
        <p>No users registered yet.</p>
        <a href="/create" class="btn btn-success">Create First Account</a>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

NOT_FOUND_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Not Found</title>""" + BASE_STYLE + """</head>
<body>
""" + NAV + """
<div class="container">
    <div class="card" style="text-align:center;">
        <h1 style="color:#e74c3c;">Profile Not Found</h1>
        <p style="margin: 15px 0;">No user found with Account ID: {{ account_id }}</p>
        <a href="/search" class="btn">Search Again</a>
        <a href="/create" class="btn btn-success">Create Account</a>
    </div>
</div>
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

        if not name or not email:
            return render_template_string(CREATE_TEMPLATE, error="Name and email are required.",
                                          name=name, email=email, phone=phone, address=address)

        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
            (name, email, phone, address)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        return redirect(url_for('view_profile', user_id=user_id, message='Account created successfully!'))

    return render_template_string(CREATE_TEMPLATE, error=None, name='', email='', phone='', address='')


@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    message = request.args.get('message', None)
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if user is None:
        return render_template_string(NOT_FOUND_TEMPLATE, account_id=user_id), 404

    return render_template_string(PROFILE_TEMPLATE, user=user, message=message)


@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    if user is None:
        conn.close()
        return render_template_string(NOT_FOUND_TEMPLATE, account_id=user_id), 404

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            user_dict = dict(user)
            user_dict['name'] = name
            user_dict['email'] = email
            user_dict['phone'] = phone
            user_dict['address'] = address
            conn.close()
            return render_template_string(EDIT_TEMPLATE, user=user_dict, error="Name and email are required.")

        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        conn.close()

        return redirect(url_for('view_profile', user_id