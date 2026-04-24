import os
import sqlite3
import hashlib
from flask import Flask, render_template_string, request, redirect, url_for, session, g

app = Flask(__name__)
app.secret_key = 'supersecretkey123'
DATABASE = 'portal.db'

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
<h2>Company Portal - Login</h2>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
<form method="post">
    <label>Username: <input type="text" name="username"></label><br><br>
    <label>Password: <input type="password" name="password"></label><br><br>
    <input type="submit" value="Login">
</form>
<p>Don't have an account? <a href="/register">Register</a></p>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Register</title></head>
<body>
<h2>Company Portal - Register</h2>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
{% if success %}<p style="color:green">{{ success }}</p>{% endif %}
<form method="post">
    <label>Username: <input type="text" name="username"></label><br><br>
    <label>Password: <input type="password" name="password"></label><br><br>
    <input type="submit" value="Register">
</form>
<p>Already have an account? <a href="/login">Login</a></p>
</body>
</html>
'''

WELCOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Welcome</title></head>
<body>
<h2>Welcome, {{ username }}!</h2>
<p>You are logged in to the Company Portal.</p>
{% if is_admin %}
<p><a href="/admin">Admin Panel</a></p>
{% endif %}
<a href="/logout">Logout</a>
</body>
</html>
'''

ADMIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head><title>Admin Panel</title></head>
<body>
<h2>Admin Panel - All Users</h2>
<table border="1">
    <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
    {% for user in users %}
    <tr><td>{{ user[0] }}</td><td>{{ user[1] }}</td><td>{{ user[2] }}</td></tr>
    {% endfor %}
</table>
<br>
<a href="/welcome">Back to Welcome</a> | <a href="/logout">Logout</a>
</body>
</html>
'''

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        db.commit()
        admin_password = hashlib.md5('admin123'.encode()).hexdigest()
        try:
            cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                           ('admin', admin_password, 1))
            db.commit()
        except sqlite3.IntegrityError:
            pass

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        hashed = hash_password(password)
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT id, username, is_admin FROM users WHERE username=? AND password=?',
                       (username, hashed))
        user = cursor.fetchone()
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['is_admin'] = user[2]
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
        if not username or not password:
            error = 'Username and password are required.'
        else:
            hashed = hash_password(password)
            db = get_db()
            cursor = db.cursor()
            try:
                cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                               (username, hashed, 0))
                db.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
    return render_template_string(REGISTER_TEMPLATE, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE,
                                   username=session['username'],
                                   is_admin=session.get('is_admin', 0))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return 'Access denied.', 403
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT id, username, is_admin FROM users')
    users = cursor.fetchall()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)