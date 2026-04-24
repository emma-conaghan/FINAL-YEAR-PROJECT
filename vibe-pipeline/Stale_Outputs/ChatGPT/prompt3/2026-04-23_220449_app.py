from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secretkey123'

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        # Create default admin if not exists
        c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if not c.fetchone():
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                      ('admin', generate_password_hash('admin'), 1))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, password, is_admin FROM users WHERE username = ?', (username,))
        return c.fetchone()

def get_all_users():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('SELECT username, is_admin FROM users')
        return c.fetchall()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            return "Username and password required", 400
        if get_user(username):
            return "User already exists", 400
        hashed_pw = generate_password_hash(password)
        with sqlite3.connect(DB) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
            conn.commit()
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Register</h2>
        <form method="POST">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <input type="submit" value="Register">
        </form>
        <a href="{{ url_for('login') }}">Login</a>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        return "Invalid credentials", 400
    return render_template_string('''
        <h2>Login</h2>
        <form method="POST">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <input type="submit" value="Login">
        </form>
        <a href="{{ url_for('register') }}">Register</a>
    ''')

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Welcome, {{ username }}!</h2>
        {% if is_admin %}
            <p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
        {% endif %}
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return "Access denied", 403
    users = get_all_users()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <ul>
        {% for user, is_admin in users %}
            <li>{{ user }} {% if is_admin %}(Admin){% endif %}</li>
        {% endfor %}
        </ul>
        <p><a href="{{ url_for('welcome') }}">Back</a></p>
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)