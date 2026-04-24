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
    .btn-blue { background: #2196F3; }
    .btn-blue:hover { background: #1976D2; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .flash { padding: 10px; margin-bottom: 15px; border-radius: 4px; }
    .flash.success { background: #d4edda; color: #155724; }
    .flash.error { background: #f8d7da; color: #721c24; }
    .field { margin-bottom: 8px; }
    .label { font-weight: bold; color: #555; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 15px; color: #2196F3; text-decoration: none; }
    nav a:hover { text-decoration: underline; }
    label { font-weight: bold; color: #555; }
</style>
'''

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>User Profile App</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>User Profile App</h1>
    <nav>
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/view">View Profile by ID</a>
    </nav>
    {% for msg, category in messages %}
        <div class="flash {{ category }}">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <h2>Welcome</h2>
        <p>Create an account, update your profile, or view a profile by account ID.</p>
        <a href="/register" class="btn">Create Account</a>
        &nbsp;
        <a href="/view" class="btn btn-blue">View Profile by ID</a>
    </div>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Register</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>Create Account</h1>
    <nav>
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/view">View Profile by ID</a>
    </nav>
    {% for msg, category in messages %}
        <div class="flash {{ category }}">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <form method="POST">
            <label>Username *</label>
            <input type="text" name="username" required value="{{ form.get('username', '') }}">
            <label>Password *</label>
            <input type="password" name="password" required>
            <label>Full Name</label>
            <input type="text" name="name" value="{{ form.get('name', '') }}">
            <label>Email</label>
            <input type="email" name="email" value="{{ form.get('email', '') }}">
            <label>Phone Number</label>
            <input type="text" name="phone" value="{{ form.get('phone', '') }}">
            <label>Address</label>
            <textarea name="address" rows="3">{{ form.get('address', '') }}</textarea>
            <button type="submit">Create Account</button>
        </form>
    </div>
</body>
</html>
'''

UPDATE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Update Profile</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>Update Profile</h1>
    <nav>
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/view">View Profile by ID</a>
    </nav>
    {% for msg, category in messages %}
        <div class="flash {{ category }}">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <p><strong>Account ID:</strong> {{ user['id'] }} &nbsp; <strong>Username:</strong> {{ user['username'] }}</p>
        <form method="POST">
            <label>Full Name</label>
            <input type="text" name="name" value="{{ user['name'] or '' }}">
            <label>Email</label>
            <input type="email" name="email" value="{{ user['email'] or '' }}">
            <label>Phone Number</label>
            <input type="text" name="phone" value="{{ user['phone'] or '' }}">
            <label>Address</label>
            <textarea name="address" rows="3">{{ user['address'] or '' }}</textarea>
            <button type="submit">Update Profile</button>
        </form>
        <br>
        <a href="/profile/{{ user['id'] }}" class="btn btn-blue">View My Profile</a>
    </div>
</body>
</html>
'''

PROFILE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Profile - {{ user['username'] }}</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>Profile Details</h1>
    <nav>
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/view">View Profile by ID</a>
    </nav>
    {% for msg, category in messages %}
        <div class="flash {{ category }}">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <div class="field"><span class="label">Account ID:</span> {{ user['id'] }}</div>
        <div class="field"><span class="label">Username:</span> {{ user['username'] }}</div>
        <div class="field"><span class="label">Full Name:</span> {{ user['name'] or 'Not provided' }}</div>
        <div class="field"><span class="label">Email:</span> {{ user['email'] or 'Not provided' }}</div>
        <div class="field"><span class="label">Phone:</span> {{ user['phone'] or 'Not provided' }}</div>
        <div class="field"><span class="label">Address:</span> {{ user['address'] or 'Not provided' }}</div>
    </div>
    <a href="/update/{{ user['id'] }}" class="btn">Update This Profile</a>
</body>
</html>
'''

VIEW_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>View Profile by ID</title>''' + BASE_STYLE + '''</head>
<body>
    <h1>View Profile by Account ID</h1>
    <nav>
        <a href="/">Home</a>
        <a href="/register">Register</a>
        <a href="/view">View Profile by ID</a>
    </nav>
    {% for msg, category in messages %}
        <div class="flash {{ category }}">{{ msg }}</div>
    {% endfor %}
    <div class="card">
        <form method="POST">
            <label>Account ID</label>
            <input type="number" name="account_id" required placeholder="Enter account ID">
            <button type="submit">View Profile</button>
        </form>
    </div>
</body>
</html>
'''

def get_flashed():
    messages = []
    if '_flashes' in app.jinja_env.globals:
        pass
    return messages

@app.route('/')
def home():
    messages = []
    return render_template_string(HOME_TEMPLATE, messages=messages)

@app.route('/register', methods=['GET', 'POST'])
def register():
    messages = []
    form_data = {}
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        form_data = {'username': username, 'name': name, 'email': email, 'phone': phone, 'address': address}
        if not username or not password:
            messages.append(('Username and password are required.', 'error'))
        else:
            try:
                conn = get_db()
                conn.execute(
                    'INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                    (username, password, name, email, phone, address)
                )
                conn.commit()
                user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
                conn.close()
                user_id = user['id']
                messages.append((f'Account created successfully! Your Account ID is {user_id}. You can now update your profile.', 'success'))
                return render_template_string(
                    REGISTER_TEMPLATE + f'<script>window.location="/update/{user_id}"</script>',
                    messages=messages, form=form_data
                )
            except sqlite3.IntegrityError:
                messages.append(('Username already exists. Please choose another.', 'error'))
            except Exception as e:
                messages.append((f'Error: {str(e)}', 'error'))
    return render_template_string(REGISTER_TEMPLATE, messages=messages, form=form_data)

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update(user_id):
    messages = []
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if not user:
        messages.append(('User not found.', 'error'))
        return render_template_string(VIEW_TEMPLATE, messages=messages)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        try:
            conn = get_db()
            conn.execute(
                'UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                (name, email, phone, address, user_id)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
            conn.close()
            messages.append(('Profile updated successfully!', 'success'))
        except Exception as e:
            messages.append((f'Error: {str(e)}', 'error'))
    return render_template_string(UPDATE_TEMPLATE, messages=messages, user=user)

@app.route('/profile/<int:user_id>')
def profile(user_id):
    messages = []
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if not user:
        messages.append(('User not found.', 'error'))
        return render_template_string(VIEW_TEMPLATE, messages=messages)
    return render_template_string(PROFILE_TEMPLATE, messages=messages, user=user)

@app.route('/view', methods=['GET', 'POST'])
def view():
    messages = []
    if request.method == 'POST':
        account_id = request.form.get('account_id', '').strip()
        try:
            account_id = int(account_id)
            conn = get_db()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (account_id,)).fetchone()
            conn.close()
            if user:
                return redirect(url_for('profile', user_id=account_id))
            else:
                messages.append(('No user found with that Account ID.', 'error'))
        except ValueError:
            messages.append(('Please enter a valid numeric Account ID.', 'error'))
        except Exception as e:
            messages.append((f'Error: {str(e)}', 'error'))
    return render_template_string(VIEW_TEMPLATE, messages=messages)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)