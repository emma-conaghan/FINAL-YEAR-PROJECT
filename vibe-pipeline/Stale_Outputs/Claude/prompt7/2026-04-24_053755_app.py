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
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                name TEXT,
                email TEXT,
                phone TEXT,
                address TEXT
            )
        ''')
        conn.commit()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    input[type=text], input[type=password], input[type=email], textarea {
        width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px;
    }
    input[type=submit] {
        background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;
    }
    input[type=submit]:hover { background: #45a049; }
    a { color: #4CAF50; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .flash { background: #ffdddd; padding: 10px; border-radius: 4px; margin-bottom: 10px; color: #a00; }
    .flash.success { background: #ddffdd; color: #080; }
    label { font-weight: bold; }
    .nav { margin-bottom: 20px; }
    .nav a { margin-right: 15px; }
    .field { margin-bottom: 10px; }
    .field span { font-weight: bold; }
</style>
"""

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>User Profile App</title>""" + BASE_STYLE + """</head>
<body>
<div class="card">
    <h1>User Profile App</h1>
    <div class="nav">
        <a href="/register">Register</a>
        <a href="/login">Login</a>
        <a href="/view">View Profile by ID</a>
    </div>
    <p>Welcome! Please register or login to manage your profile.</p>
</div>
</body>
</html>
"""

REGISTER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Register</title>""" + BASE_STYLE + """</head>
<body>
<div class="card">
    <h2>Register</h2>
    {% for msg in messages %}
    <div class="flash">{{ msg }}</div>
    {% endfor %}
    <form method="POST">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Register">
    </form>
    <p>Already have an account? <a href="/login">Login</a></p>
</div>
</body>
</html>
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Login</title>""" + BASE_STYLE + """</head>
<body>
<div class="card">
    <h2>Login</h2>
    {% for msg in messages %}
    <div class="flash">{{ msg }}</div>
    {% endfor %}
    <form method="POST">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Login">
    </form>
    <p>No account? <a href="/register">Register</a></p>
</div>
</body>
</html>
"""

PROFILE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Edit Profile</title>""" + BASE_STYLE + """</head>
<body>
<div class="card">
    <h2>Edit Profile (Account ID: {{ user['id'] }})</h2>
    {% for msg in messages %}
    <div class="flash {{ 'success' if 'updated' in msg else '' }}">{{ msg }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" value="{{ user['name'] or '' }}">
        <label>Email:</label>
        <input type="email" name="email" value="{{ user['email'] or '' }}">
        <label>Phone:</label>
        <input type="text" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address:</label>
        <input type="text" name="address" value="{{ user['address'] or '' }}">
        <input type="submit" value="Update Profile">
    </form>
    <p><a href="/logout">Logout</a> | <a href="/view?id={{ user['id'] }}">View Public Profile</a></p>
</div>
</body>
</html>
"""

VIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>View Profile</title>""" + BASE_STYLE + """</head>
<body>
<div class="card">
    <h2>View Profile by Account ID</h2>
    {% for msg in messages %}
    <div class="flash">{{ msg }}</div>
    {% endfor %}
    <form method="GET">
        <label>Account ID:</label>
        <input type="text" name="id" value="{{ search_id or '' }}">
        <input type="submit" value="Search">
    </form>
    {% if user %}
    <hr>
    <h3>Profile Details</h3>
    <div class="field"><span>Account ID:</span> {{ user['id'] }}</div>
    <div class="field"><span>Username:</span> {{ user['username'] }}</div>
    <div class="field"><span>Name:</span> {{ user['name'] or 'Not provided' }}</div>
    <div class="field"><span>Email:</span> {{ user['email'] or 'Not provided' }}</div>
    <div class="field"><span>Phone:</span> {{ user['phone'] or 'Not provided' }}</div>
    <div class="field"><span>Address:</span> {{ user['address'] or 'Not provided' }}</div>
    {% endif %}
    <p><a href="/">Home</a></p>
</div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    messages = []
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            messages.append('Username and password are required.')
        else:
            try:
                with get_db() as conn:
                    conn.execute(
                        'INSERT INTO users (username, password) VALUES (?, ?)',
                        (username, password)
                    )
                    conn.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                messages.append('Username already exists.')
    return render_template_string(REGISTER_TEMPLATE, messages=messages)

@app.route('/login', methods=['GET', 'POST'])
def login():
    messages = []
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        with get_db() as conn:
            user = conn.execute(
                'SELECT * FROM users WHERE username=? AND password=?',
                (username, password)
            ).fetchone()
        if user:
            from flask import session
            session['user_id'] = user['id']
            return redirect(url_for('profile'))
        else:
            messages.append('Invalid username or password.')
    return render_template_string(LOGIN_TEMPLATE, messages=messages)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    from flask import session
    messages = []
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    if not user:
        return redirect(url_for('login'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        with get_db() as conn:
            conn.execute(
                'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                (name, email, phone, address, user_id)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
        messages.append('Profile updated successfully.')
    return render_template_string(PROFILE_TEMPLATE, user=user, messages=messages)

@app.route('/logout')
def logout():
    from flask import session
    session.clear()
    return redirect(url_for('home'))

@app.route('/view', methods=['GET'])
def view():
    messages = []
    user = None
    search_id = request.args.get('id', '').strip()
    if search_id:
        try:
            uid = int(search_id)
            with get_db() as conn:
                user = conn.execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone()
            if not user:
                messages.append('No user found with that ID.')
        except ValueError:
            messages.append('Please enter a valid numeric ID.')
    return render_template_string(VIEW_TEMPLATE, user=user, messages=messages, search_id=search_id)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)