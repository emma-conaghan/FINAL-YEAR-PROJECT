from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      is_admin INTEGER NOT NULL DEFAULT 0)''')
        # Create a default admin if not exists
        c.execute("SELECT * FROM users WHERE username='admin'")
        if c.fetchone() is None:
            pw_hash = hashlib.sha256('adminpass'.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', pw_hash, 1))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_admin FROM users WHERE username=?", (username,))
        return c.fetchone()

def add_user(username, password):
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, pw_hash))
        conn.commit()

def get_all_users():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, is_admin FROM users")
        return c.fetchall()

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            error = 'Username and password are required.'
        elif get_user(username):
            error = 'Username already exists.'
        else:
            try:
                add_user(username, password)
                return redirect(url_for('login'))
            except Exception:
                error = 'Error during registration.'
    return render_template_string('''
        <h2>Register</h2>
        {% if error %}<p style="color:red">{{ error }}</p>{% endif %}
        <form method="post">
            Username: <input type="text" name="username"><br>
            Password: <input type="password" name="password"><br>
            <input type="submit" value="Register">
        </form>
        <p>Already registered? <a href="{{ url_for('login') }}">Login here</a>.</p>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = get_user(username)
        if user:
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            if pw_hash == user[2]:
                session['user'] = user[1]
                session['is_admin'] = bool(user[3])
                return redirect(url_for('welcome'))
        error = "Invalid username or password."
    return render_template_string('''
        <h2>Login</h2>
        {% if error %}<p style="color:red">{{ error }}</p>{% endif %}
        <form method="post">
            Username: <input type="text" name="username"><br>
            Password: <input type="password" name="password"><br>
            <input type="submit" value="Login">
        </form>
        <p>New user? <a href="{{ url_for('register') }}">Register here</a>.</p>
    ''', error=error)

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Welcome, {{ user }}!</h2>
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
        {% if is_admin %}
            <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
        {% endif %}
    ''', user=session['user'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'user' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    users = get_all_users()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <table border="1" cellpadding="5">
            <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
            {% for u in users %}
            <tr>
                <td>{{ u[0] }}</td>
                <td>{{ u[1] }}</td>
                <td>{{ 'Yes' if u[2] else 'No' }}</td>
            </tr>
            {% endfor %}
        </table>
        <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)