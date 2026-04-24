from flask import Flask, request, redirect, url_for, render_template_string, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this in production

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users
                       (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0)''')
        # Insert default admin if not exists
        cur.execute("SELECT * FROM users WHERE username='admin'")
        if not cur.fetchone():
            cur.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                        ('admin', generate_password_hash('admin'), 1))
        con.commit()

def get_user(username):
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("SELECT id, username, password, is_admin FROM users WHERE username=?", (username,))
        return cur.fetchone()

def get_all_users():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("SELECT username, is_admin FROM users")
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
        password = request.form.get('password', '').strip()
        if not username or not password:
            return "Username and password required", 400
        try:
            with sqlite3.connect(DB) as con:
                cur = con.cursor()
                cur.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                            (username, generate_password_hash(password)))
                con.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already taken", 400
    return render_template_string('''
        <h2>Register</h2>
        <form method="POST">
            Username: <input name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <button type="submit">Register</button>
        </form>
        <a href="{{ url_for('login') }}">Login</a>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        return "Invalid credentials", 401
    return render_template_string('''
        <h2>Login</h2>
        <form method="POST">
            Username: <input name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <button type="submit">Login</button>
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
        <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
        {% endif %}
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return "Unauthorized", 403
    users = get_all_users()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <ul>
        {% for user, is_admin in users %}
            <li>{{ user }} {% if is_admin %}(Admin){% endif %}</li>
        {% endfor %}
        </ul>
        <p><a href="{{ url_for('welcome') }}">Back</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)