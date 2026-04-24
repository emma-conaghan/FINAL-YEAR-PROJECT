from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
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
        c.execute('SELECT * FROM users WHERE username=?', ('admin',))
        if c.fetchone() is None:
            admin_hashed = generate_password_hash('adminpass')
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?,?,1)', ('admin', admin_hashed))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, password, is_admin FROM users WHERE username=?', (username,))
        return c.fetchone()

def get_all_users():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('SELECT username, is_admin FROM users')
        return c.fetchall()

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            return 'Username and password required', 400
        if get_user(username) is not None:
            return 'User already exists', 400
        hashed = generate_password_hash(password)
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
            conn.commit()
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Register</h2>
        <form method="post">
          Username: <input name="username" required><br>
          Password: <input name="password" type="password" required><br>
          <input type="submit" value="Register">
        </form>
        <a href="{{ url_for('login') }}">Login</a>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = get_user(username)
        if user is None or not check_password_hash(user[2], password):
            return 'Invalid credentials', 401
        session['user'] = {'id': user[0], 'username': user[1], 'is_admin': user[3]}
        return redirect(url_for('welcome'))
    return render_template_string('''
        <h2>Login</h2>
        <form method="post">
          Username: <input name="username" required><br>
          Password: <input name="password" type="password" required><br>
          <input type="submit" value="Login">
        </form>
        <a href="{{ url_for('register') }}">Register</a>
    ''')

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    return render_template_string('''
        <h2>Welcome {{username}}</h2>
        {% if is_admin %}
          <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
        {% endif %}
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', username=user['username'], is_admin=user['is_admin'])

@app.route('/admin')
def admin():
    if 'user' not in session or not session['user']['is_admin']:
        return 'Access denied', 403
    users = get_all_users()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <table border=1>
          <tr><th>Username</th><th>Admin</th></tr>
          {% for username, is_admin in users %}
            <tr><td>{{ username }}</td><td>{{ 'Yes' if is_admin else 'No' }}</td></tr>
          {% endfor %}
        </table>
        <p><a href="{{ url_for('welcome') }}">Back</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)