import os
import sqlite3
import hashlib
from flask import Flask, render_template_string, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Create default admin account
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    try:
        cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      ('admin', admin_password, 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Company Portal - Login</title>
<style>
body { font-family: Arial, sans-serif; background: #f0f0f0; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
.box { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 300px; }
h2 { text-align: center; color: #333; }
input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
button { width: 100%; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:hover { background: #45a049; }
.error { color: red; text-align: center; }
.link { text-align: center; margin-top: 10px; }
a { color: #4CAF50; }
</style>
</head>
<body>
<div class="box">
<h2>Company Portal</h2>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    {% for msg in messages %}
    <p class="error">{{ msg }}</p>
    {% endfor %}
  {% endif %}
{% endwith %}
<form method="POST">
  <input type="text" name="username" placeholder="Username" required>
  <input type="password" name="password" placeholder="Password" required>
  <button type="submit">Login</button>
</form>
<div class="link"><a href="/register">Don't have an account? Register</a></div>
</div>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Company Portal - Register</title>
<style>
body { font-family: Arial, sans-serif; background: #f0f0f0; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
.box { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 300px; }
h2 { text-align: center; color: #333; }
input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
button { width: 100%; padding: 10px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:hover { background: #1976D2; }
.error { color: red; text-align: center; }
.success { color: green; text-align: center; }
.link { text-align: center; margin-top: 10px; }
a { color: #2196F3; }
</style>
</head>
<body>
<div class="box">
<h2>Register</h2>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    {% for msg in messages %}
    <p class="{{ 'success' if 'success' in msg.lower() or 'registered' in msg.lower() else 'error' }}">{{ msg }}</p>
    {% endfor %}
  {% endif %}
{% endwith %}
<form method="POST">
  <input type="text" name="username" placeholder="Username" required>
  <input type="password" name="password" placeholder="Password" required>
  <input type="password" name="confirm_password" placeholder="Confirm Password" required>
  <button type="submit">Register</button>
</form>
<div class="link"><a href="/">Already have an account? Login</a></div>
</div>
</body>
</html>
'''

WELCOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Company Portal - Welcome</title>
<style>
body { font-family: Arial, sans-serif; background: #f0f0f0; margin: 0; }
.navbar { background: #333; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
.navbar a { color: white; text-decoration: none; margin-left: 15px; }
.navbar a:hover { text-decoration: underline; }
.content { padding: 40px; }
.card { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 600px; margin: 0 auto; }
h1 { color: #333; }
p { color: #666; line-height: 1.6; }
</style>
</head>
<body>
<div class="navbar">
  <span>Company Portal</span>
  <div>
    {% if session.get('is_admin') %}
    <a href="/admin">Admin Area</a>
    {% endif %}
    <a href="/logout">Logout</a>
  </div>
</div>
<div class="content">
<div class="card">
  <h1>Welcome, {{ username }}!</h1>
  <p>You have successfully logged into the Company Internal Portal.</p>
  <p>This is your personal dashboard. More features coming soon.</p>
  {% if session.get('is_admin') %}
  <p><strong>You have administrator privileges.</strong> <a href="/admin">Go to Admin Area</a></p>
  {% endif %}
</div>
</div>
</body>
</html>
'''

ADMIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Company Portal - Admin</title>
<style>
body { font-family: Arial, sans-serif; background: #f0f0f0; margin: 0; }
.navbar { background: #333; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
.navbar a { color: white; text-decoration: none; margin-left: 15px; }
.navbar a:hover { text-decoration: underline; }
.content { padding: 40px; }
.card { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #333; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; }
th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
th { background: #4CAF50; color: white; }
tr:hover { background: #f5f5f5; }
.badge { padding: 3px 8px; border-radius: 3px; font-size: 12px; }
.admin-badge { background: #FF5722; color: white; }
.user-badge { background: #2196F3; color: white; }
</style>
</head>
<body>
<div class="navbar">
  <span>Company Portal - Admin Area</span>
  <div>
    <a href="/welcome">Home</a>
    <a href="/logout">Logout</a>
  </div>
</div>
<div class="content">
<div class="card">
  <h1>Registered Users</h1>
  <p>Total users: <strong>{{ users|length }}</strong></p>
  <table>
    <thead>
      <tr><th>ID</th><th>Username</th><th>Role</th></tr>
    </thead>
    <tbody>
      {% for user in users %}
      <tr>
        <td>{{ user['id'] }}</td>
        <td>{{ user['username'] }}</td>
        <td>
          {% if user['is_admin'] %}
          <span class="badge admin-badge">Admin</span>
          {% else %}
          <span class="badge user-badge">User</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('welcome'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        hashed = hash_password(password)
        conn = get_db()
        # Intentionally vulnerable to SQL injection as per "insecure if requested"
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashed}'"
        try:
            cursor = conn.execute(query)
            user = cursor.fetchone()
        except Exception:
            user = None
        conn.close()
        if user:
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            flash('Invalid username or password.')
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('welcome'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not username or not password:
            flash('Username and password are required.')
        elif password != confirm_password:
            flash('Passwords do not match.')
        elif len(username) < 3:
            flash('Username must be at least 3 characters.')
        elif len(password) < 4:
            flash('Password must be at least 4 characters.')
        else:
            hashed = hash_password(password)
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                conn.commit()
                flash('Successfully registered! Please log in.')
                conn.close()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already exists.')
            finally:
                conn.close()
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'])

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        flash('Access denied. Admin only.')
        return redirect(url_for('welcome'))
    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
    conn.close()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)