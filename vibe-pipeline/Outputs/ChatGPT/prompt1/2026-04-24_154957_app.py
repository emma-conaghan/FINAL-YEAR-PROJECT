from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DB_NAME = 'users.db'

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if not c.fetchone():
            c.execute('INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)',
                      ('admin', generate_password_hash('adminpass'), 1))
            conn.commit()

init_db()

def get_user(username):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, password_hash, is_admin FROM users WHERE username = ?', (username,))
        row = c.fetchone()
        if row:
            return {'id': row[0], 'username': row[1], 'password_hash': row[2], 'is_admin': bool(row[3])}
    return None

def add_user(username, password):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                      (username, generate_password_hash(password)))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('welcome'))
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Username and password are required.'
        elif get_user(username):
            error = 'Username already taken.'
        else:
            if add_user(username, password):
                return redirect(url_for('login'))
            else:
                error = 'Registration failed.'
    return render_template_string('''
    <h2>Register</h2>
    <form method="post">
      Username: <input type="text" name="username" required><br>
      Password: <input type="password" name="password" required><br>
      <input type="submit" value="Register">
    </form>
    <p style="color:red;">{{ error }}</p>
    <p>Already have an account? <a href="{{ url_for('login') }}">Login here</a>.</p>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('welcome'))
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user and check_password_hash(user['password_hash'], password):
            session['user'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('welcome'))
        error = 'Invalid username or password.'
    return render_template_string('''
    <h2>Login</h2>
    <form method="post">
      Username: <input type="text" name="username" required><br>
      Password: <input type="password" name="password" required><br>
      <input type="submit" value="Login">
    </form>
    <p style="color:red;">{{ error }}</p>
    <p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a>.</p>
    ''', error=error)

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
    <h2>Welcome, {{ user }}!</h2>
    {% if is_admin %}
    <p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
    {% endif %}
    <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', user=session['user'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'user' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('SELECT username, is_admin FROM users ORDER BY username')
        users = c.fetchall()
    return render_template_string('''
    <h2>Admin Area - Registered Users</h2>
    <table border="1" cellpadding="5" cellspacing="0">
      <tr><th>Username</th><th>Is Admin</th></tr>
      {% for username, is_admin_flag in users %}
        <tr><td>{{ username }}</td><td>{{ 'Yes' if is_admin_flag else 'No' }}</td></tr>
      {% endfor %}
    </table>
    <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)