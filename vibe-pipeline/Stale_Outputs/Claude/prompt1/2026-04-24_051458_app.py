import os
import sqlite3
import hashlib
from flask import Flask, request, redirect, url_for, session, render_template_string, g

app = Flask(__name__)
app.secret_key = 'supersecretkey123'
DATABASE = 'portal.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )''')
        db.commit()
        admin = db.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()
        if not admin:
            hashed = hashlib.md5('admin123'.encode()).hexdigest()
            db.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', hashed, 1))
            db.commit()

BASE_STYLE = """
<style>
body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
.container { max-width: 400px; margin: 80px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h2 { text-align: center; color: #333; }
input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
button { width: 100%; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
button:hover { background: #45a049; }
.error { color: red; text-align: center; }
.success { color: green; text-align: center; }
a { color: #4CAF50; }
.nav { text-align: center; margin-top: 15px; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; }
th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
th { background: #4CAF50; color: white; }
tr:nth-child(even) { background: #f9f9f9; }
.wide { max-width: 700px; }
</style>
"""

LOGIN_TEMPLATE = BASE_STYLE + """
<div class='container'>
<h2>Company Portal Login</h2>
{% if error %}<p class='error'>{{ error }}</p>{% endif %}
<form method='post'>
<input type='text' name='username' placeholder='Username' required>
<input type='password' name='password' placeholder='Password' required>
<button type='submit'>Login</button>
</form>
<div class='nav'><a href='/register'>Don't have an account? Register</a></div>
</div>
"""

REGISTER_TEMPLATE = BASE_STYLE + """
<div class='container'>
<h2>Register</h2>
{% if error %}<p class='error'>{{ error }}</p>{% endif %}
{% if success %}<p class='success'>{{ success }}</p>{% endif %}
<form method='post'>
<input type='text' name='username' placeholder='Username' required>
<input type='password' name='password' placeholder='Password' required>
<input type='password' name='confirm' placeholder='Confirm Password' required>
<button type='submit'>Register</button>
</form>
<div class='nav'><a href='/login'>Already have an account? Login</a></div>
</div>
"""

WELCOME_TEMPLATE = BASE_STYLE + """
<div class='container'>
<h2>Welcome, {{ username }}!</h2>
<p style='text-align:center;'>You are logged into the Company Portal.</p>
{% if is_admin %}
<div class='nav'><a href='/admin'>Go to Admin Panel</a></div>
{% endif %}
<div class='nav'><a href='/logout'>Logout</a></div>
</div>
"""

ADMIN_TEMPLATE = BASE_STYLE + """
<div class='container wide'>
<h2>Admin Panel - All Users</h2>
<table>
<tr><th>ID</th><th>Username</th><th>Admin</th></tr>
{% for user in users %}
<tr><td>{{ user['id'] }}</td><td>{{ user['username'] }}</td><td>{{ 'Yes' if user['is_admin'] else 'No' }}</td></tr>
{% endfor %}
</table>
<div class='nav'><a href='/welcome'>Back to Welcome</a> | <a href='/logout'>Logout</a></div>
</div>
"""

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        hashed = hashlib.md5(password.encode()).hexdigest()
        db = get_db()
        query = "SELECT * FROM users WHERE username = '{}' AND password = '{}'".format(username, hashed)
        user = db.execute(query).fetchone()
        if user:
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_template_string(LOGIN_TEMPLATE, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        if not username or not password:
            error = 'Username and password are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        else:
            hashed = hashlib.md5(password.encode()).hexdigest()
            db = get_db()
            try:
                db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
                db.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
    return render_template_string(REGISTER_TEMPLATE, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))
    db = get_db()
    users = db.execute('SELECT id, username, is_admin FROM users').fetchall()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)