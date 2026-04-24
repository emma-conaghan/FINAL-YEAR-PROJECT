from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 's3cr3t_key'

DATABASE = 'app.db'

def init_db():
    with sqlite3.connect(DATABASE) as con:
        cur = con.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        con.commit()
        # Create default admin user if not exists
        cur.execute('SELECT * FROM users WHERE username=?', ('admin',))
        if not cur.fetchone():
            cur.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', 
                        ('admin', generate_password_hash('admin'), 1))
            con.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as con:
        cur = con.cursor()
        cur.execute('SELECT id, username, password, is_admin FROM users WHERE username = ?', (username,))
        return cur.fetchone()

def get_all_users():
    with sqlite3.connect(DATABASE) as con:
        cur = con.cursor()
        cur.execute('SELECT username, is_admin FROM users')
        return cur.fetchall()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return "Username and password are required", 400
        if get_user(username):
            return "User already exists", 400
        hashed_pw = generate_password_hash(password)
        with sqlite3.connect(DATABASE) as con:
            cur = con.cursor()
            cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
            con.commit()
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Register</h2>
        <form method="POST">
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
        password = request.form.get('password', '')
        user = get_user(username)
        if not user or not check_password_hash(user[2], password):
            return "Invalid credentials", 400
        session['username'] = user[1]
        session['is_admin'] = bool(user[3])
        return redirect(url_for('welcome'))
    return render_template_string('''
        <h2>Login</h2>
        <form method="POST">
            Username: <input name="username" required><br>
            Password: <input name="password" type="password" required><br>
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
            <a href="{{ url_for('admin') }}">Admin Area</a><br>
        {% endif %}
        <a href="{{ url_for('logout') }}">Logout</a>
    ''', username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return "Access denied", 403
    users = get_all_users()
    return render_template_string('''
        <h2>Admin Area - All Registered Users</h2>
        <ul>
        {% for username, is_admin in users %}
            <li>{{ username }} {% if is_admin %}(Admin){% endif %}</li>
        {% endfor %}
        </ul>
        <a href="{{ url_for('welcome') }}">Back</a>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    init_db()
    app.run(debug=True)