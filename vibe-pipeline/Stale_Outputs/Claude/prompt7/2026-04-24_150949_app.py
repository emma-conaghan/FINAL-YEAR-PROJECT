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
            email TEXT,
            phone TEXT,
            address TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 0 20px; background: #f4f4f4; }
    h1, h2 { color: #333; }
    input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background: #45a049; }
    a { color: #4CAF50; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .flash { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
    .flash.success { background: #d4edda; color: #155724; }
    .field-label { font-weight: bold; color: #555; }
    .nav { margin-bottom: 20px; }
    .nav a { margin-right: 15px; }
</style>
'''

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>User Profile App</title>''' + BASE_STYLE + '''</head>
<body>
<div class="card">
    <h1>User Profile App</h1>
    <p>Manage your account and profile information.</p>
    <div class="nav">
        <a href="/register">Create Account</a>
        <a href="/login">Login</a>
        <a href="/view">View Profile by ID</a>
    </div>
</div>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Create Account</title>''' + BASE_STYLE + '''</head>
<body>
<div class="card">
    <div class="nav"><a href="/">Home</a></div>
    <h2>Create Account</h2>
    {% for msg in messages %}
    <div class="flash {{ msg.category }}">{{ msg.message }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" required>
        <label>Email:</label>
        <input type="email" name="email" required>
        <label>Phone:</label>
        <input type="text" name="phone">
        <label>Address:</label>
        <textarea name="address" rows="3"></textarea>
        <label>Password:</label>
        <input type="password" name="password" required>
        <button type="submit">Create Account</button>
    </form>
</div>
</body>
</html>
'''

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Login</title>''' + BASE_STYLE + '''</head>
<body>
<div class="card">
    <div class="nav"><a href="/">Home</a></div>
    <h2>Login</h2>
    {% for msg in messages %}
    <div class="flash {{ msg.category }}">{{ msg.message }}</div>
    {% endfor %}
    <form method="POST">
        <label>Email:</label>
        <input type="email" name="email" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <button type="submit">Login</button>
    </form>
</div>
</body>
</html>
'''

PROFILE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Profile - {{ user['name'] }}</title>''' + BASE_STYLE + '''</head>
<body>
<div class="card">
    <div class="nav"><a href="/">Home</a></div>
    <h2>Profile: {{ user['name'] }}</h2>
    {% for msg in messages %}
    <div class="flash {{ msg.category }}">{{ msg.message }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" value="{{ user['name'] }}" required>
        <label>Email:</label>
        <input type="email" name="email" value="{{ user['email'] }}" required>
        <label>Phone:</label>
        <input type="text" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address:</label>
        <textarea name="address" rows="3">{{ user['address'] or '' }}</textarea>
        <button type="submit">Update Profile</button>
    </form>
    <br>
    <p><span class="field-label">Account ID:</span> {{ user['id'] }}</p>
</div>
</body>
</html>
'''

VIEW_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>View Profile</title>''' + BASE_STYLE + '''</head>
<body>
<div class="card">
    <div class="nav"><a href="/">Home</a></div>
    <h2>View Profile by Account ID</h2>
    {% for msg in messages %}
    <div class="flash {{ msg.category }}">{{ msg.message }}</div>
    {% endfor %}
    <form method="POST">
        <label>Account ID:</label>
        <input type="number" name="account_id" required>
        <button type="submit">View Profile</button>
    </form>
    {% if user %}
    <hr>
    <h3>Profile Details</h3>
    <p><span class="field-label">ID:</span> {{ user['id'] }}</p>
    <p><span class="field-label">Name:</span> {{ user['name'] }}</p>
    <p><span class="field-label">Email:</span> {{ user['email'] }}</p>
    <p><span class="field-label">Phone:</span> {{ user['phone'] or 'N/A' }}</p>
    <p><span class="field-label">Address:</span> {{ user['address'] or 'N/A' }}</p>
    {% endif %}
</div>
</body>
</html>
'''

class FlashMessage:
    def __init__(self, message, category='flash'):
        self.message = message
        self.category = category

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    messages = []
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        password = request.form.get('password', '').strip()

        if not name or not email or not password:
            messages.append(FlashMessage('Name, email, and password are required.'))
        else:
            conn = get_db()
            existing = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
            if existing:
                messages.append(FlashMessage('An account with that email already exists.'))
                conn.close()
            else:
                cursor = conn.execute(
                    'INSERT INTO users (name, email, phone, address, password) VALUES (?, ?, ?, ?, ?)',
                    (name, email, phone, address, password)
                )
                conn.commit()
                new_id = cursor.lastrowid
                conn.close()
                messages.append(FlashMessage(f'Account created! Your Account ID is {new_id}. <a href="/profile/{new_id}">Go to Profile</a>', 'success'))

    return render_template_string(REGISTER_TEMPLATE, messages=messages)

@app.route('/login', methods=['GET', 'POST'])
def login():
    messages = []
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password)).fetchone()
        conn.close()

        if user:
            return redirect(url_for('profile', user_id=user['id']))
        else:
            messages.append(FlashMessage('Invalid email or password.'))

    return render_template_string(LOGIN_TEMPLATE, messages=messages)

@app.route('/profile/<int:user_id>', methods=['GET', 'POST'])
def profile(user_id):
    messages = []
    conn = get_db()

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not name or not email:
            messages.append(FlashMessage('Name and email are required.'))
        else:
            conn.execute(
                'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
                (name, email, phone, address, user_id)
            )
            conn.commit()
            messages.append(FlashMessage('Profile updated successfully!', 'success'))

    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    if not user:
        return '<h2>User not found.</h2><a href="/">Home</a>', 404

    return render_template_string(PROFILE_TEMPLATE, user=user, messages=messages)

@app.route('/view', methods=['GET', 'POST'])
def view():
    messages = []
    user = None

    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        if not account_id.isdigit():
            messages.append(FlashMessage('Please enter a valid numeric Account ID.'))
        else:
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (int(account_id),)).fetchone()
            conn.close()
            if not user:
                messages.append(FlashMessage(f'No account found with ID {account_id}.'))

    return render_template_string(VIEW_TEMPLATE, messages=messages, user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)