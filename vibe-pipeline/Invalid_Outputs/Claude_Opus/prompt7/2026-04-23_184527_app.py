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
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        nav {
            background-color: #333;
            padding: 10px 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        nav a {
            color: white;
            text-decoration: none;
            margin-right: 20px;
            font-size: 16px;
        }
        nav a:hover {
            text-decoration: underline;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
        }
        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
            color: #555;
        }
        input[type="text"], input[type="email"], input[type="tel"], textarea, input[type="number"] {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 14px;
        }
        textarea {
            height: 80px;
            resize: vertical;
        }
        button, input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 20px;
        }
        button:hover, input[type="submit"]:hover {
            background-color: #45a049;
        }
        .profile-card {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            margin-top: 15px;
        }
        .profile-card p {
            margin: 8px 0;
            font-size: 15px;
        }
        .profile-card .field-label {
            font-weight: bold;
            color: #555;
        }
        .message {
            padding: 10px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .user-list {
            list-style: none;
            padding: 0;
        }
        .user-list li {
            padding: 10px 15px;
            border: 1px solid #ddd;
            margin-bottom: 5px;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .user-list li a {
            color: #4CAF50;
            text-decoration: none;
        }
        .user-list li a:hover {
            text-decoration: underline;
        }
        .btn-edit {
            background-color: #2196F3;
        }
        .btn-edit:hover {
            background-color: #1976D2;
        }
        .btn-small {
            padding: 6px 12px;
            font-size: 13px;
            margin-top: 0;
        }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/create">Create Account</a>
        <a href="/lookup">View Profile</a>
        <a href="/users">All Users</a>
    </nav>
    <div class="container">
        {{ content }}
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    content = '''
        <h1>Welcome to User Profile Manager</h1>
        <p>Use this application to create and manage user profiles.</p>
        <p><a href="/create"><button>Create New Account</button></a></p>
        <p><a href="/lookup"><button style="background-color:#2196F3;">View a Profile</button></a></p>
        <p><a href="/users"><button style="background-color:#FF9800;">View All Users</button></a></p>
    '''
    return render_template_string(BASE_TEMPLATE, title="Home", content=content)

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            content = '''
                <h1>Create Account</h1>
                <div class="message error">Name and Email are required fields.</div>
                <form method="POST">
                    <label>Name *</label>
                    <input type="text" name="name" value="{{ name }}" required>
                    <label>Email *</label>
                    <input type="email" name="email" value="{{ email }}" required>
                    <label>Phone</label>
                    <input type="tel" name="phone" value="{{ phone }}">
                    <label>Address</label>
                    <textarea name="address">{{ address }}</textarea>
                    <br>
                    <input type="submit" value="Create Account">
                </form>
            '''
            return render_template_string(BASE_TEMPLATE, title="Create Account", content=content,
                                         name=name, email=email, phone=phone, address=address)

        conn = get_db()
        cursor = conn.execute(
            'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
            (name, email, phone, address)
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return redirect(url_for('view_profile', user_id=user_id, created='1'))

    content = '''
        <h1>Create Account</h1>
        <form method="POST">
            <label>Name *</label>
            <input type="text" name="name" required>
            <label>Email *</label>
            <input type="email" name="email" required>
            <label>Phone</label>
            <input type="tel" name="phone">
            <label>Address</label>
            <textarea name="address"></textarea>
            <br>
            <input type="submit" value="Create Account">
        </form>
    '''
    return render_template_string(BASE_TEMPLATE, title="Create Account", content=content)

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        content = '''
            <h1>Profile Not Found</h1>
            <div class="message error">No user found with ID {{ user_id }}.</div>
            <a href="/lookup"><button>Try Again</button></a>
        '''
        return render_template_string(BASE_TEMPLATE, title="Not Found", content=content, user_id=user_id), 404

    created = request.args.get('created')
    updated = request.args.get('updated')

    message = ''
    if created:
        message = '<div class="message success">Account created successfully! Your Account ID is <strong>{}</strong>.</div>'.format(user['id'])
    elif updated:
        message = '<div class="message success">Profile updated successfully!</div>'

    content = '''
        <h1>User Profile</h1>
        ''' + message + '''
        <div class="profile-card">
            <p><span class="field-label">Account ID:</span> {{ user.id }}</p>
            <p><span class="field-label">Name:</span> {{ user.name }}</p>
            <p><span class="field-label">Email:</span> {{ user.email }}</p>
            <p><span class="field-label">Phone:</span> {{ user.phone if user.phone else 'N/A' }}</p>
            <p><span class="field-label">Address:</span> {{ user.address if user.address else 'N/A' }}</p>
        </div>
        <a href="/edit/{{ user.id }}"><button class="btn-edit">Edit Profile</button></a>
    '''
    return render_template_string(BASE_TEMPLATE, title="Profile - " + user['name'], content=content, user=user)

@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

    if not user:
        conn.close()
        content = '''
            <h1>User Not Found</h1>
            <div class="message error">No user found with ID {{ user_id }}.</div>
            <a href="/lookup"><button>Try Again</button></a>
        '''
        return render_template_string(BASE_TEMPLATE, title="Not Found", content=content, user_id=user_id), 404

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            content = '''
                <h1>Edit Profile (ID: {{ user_id }})</h1>
                <div class="message error">Name and Email are required fields.</div>
                <form method="POST">
                    <label>Name *</label>
                    <input type="text" name="name" value="{{ name }}" required>
                    <label>Email *</label>
                    <input type="email" name="email" value="{{ email }}" required>
                    <label>Phone</label>
                    <input type="tel" name="phone" value="{{ phone }}">
                    <label>Address</label>
                    <textarea name="address">{{ address }}</textarea>
                    <br>
                    <input type="submit" value="Update Profile">
                </form>
            '''
            conn.close()
            return render_template_string(BASE_TEMPLATE, title="Edit Profile", content=content,
                                         user_id=user_id, name=name, email=email, phone=phone, address=address)

        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        conn.close()

        return redirect(url_for('view_profile', user_id=user_id, updated='1'))

    content = '''
        <h1>Edit Profile (ID: {{ user.id }})</h1>
        <form method="POST">
            <label>Name *</label>
            <input type="text" name="name" value="{{ user.name }}" required>
            <label>Email *</label>
            <input type="email" name="email" value="{{ user.email }}" required>
            <label>Phone</label>
            <input type="tel" name="phone" value="{{ user.phone if user.phone else '' }}">
            <label>Address</label>
            <textarea name="address">{{ user.address if user.address else '' }}</textarea>
            <br>
            <input type="submit" value="Update Profile">
            <a href="/profile/{{ user.id }}"><button type="button" style="background-color:#999;">Cancel</button></a>
        </form>
    '''
    conn.close()
    return render_template_string(BASE_TEMPLATE, title="Edit Profile - " + user['name'], content=content, user=user)

@app.route('/lookup', methods=['GET', 'POST'])
def lookup():
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        if account_id and account_id.isdigit():
            return redirect(url_for('view_profile', user_id=int(account_id)))
        else:
            content = '''
                <h1>View Profile by Account ID</h1>
                <div class="message error">Please enter a valid numeric Account ID.</div>
                <form method="POST">
                    <label>Account ID</label>
                    <input type="number" name="account_id" min="1" required>
                    <br>
                    <input type="submit" value="View Profile">
                </form>
            '''
            return render_template_string(BASE_TEMPLATE, title="View Profile", content=content)

    content = '''
        <h1>View Profile by Account ID</h1>
        <form method="POST">
            <label>Account ID</label>
            <input type="number" name="account_id" min="1" required>
            <br>
            <input type="submit" value="View Profile">
        </form>
    '''
    return render_template_string(BASE_TEMPLATE, title="View Profile", content=content)

@app.route('/users')
def list_users():
    conn = get_db()
    users = conn.execute('SELECT id, name, email FROM users ORDER BY id DESC').fetchall()
    conn.close()

    if not users:
        content = '''
            <h1>All Users</h1>
            <p>No users found. <a href="/create">Create the first account!</a></p>
        '''
    else:
        user_items = ''
        for u in users:
            user_items += '<li><span><strong>{}</strong> - {}</span> <span><a href="/profile/{}">View</a> | <a href="/edit/{}">Edit</a></span></li>'.format(
                u['name'], u['email'], u['id'], u['id']
            )
        content = '''
            <h1>All Users</h1>
            <p>Total users: ''' + str(len(users)) + '''</p>
            <ul class="user-list">
                ''' + user_items + '''
            </ul>
        '''
    return render_template_string(BASE_TEMPLATE