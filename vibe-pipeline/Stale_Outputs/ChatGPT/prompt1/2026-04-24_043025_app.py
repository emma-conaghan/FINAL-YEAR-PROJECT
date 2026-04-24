from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'replace_with_a_secret_key'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0
        )
    ''')
    # Create default admin if not exists
    cur.execute("SELECT * FROM users WHERE username=?", ('admin',))
    if not cur.fetchone():
        admin_pass = hashlib.sha256('adminpass'.encode()).hexdigest()
        cur.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', admin_pass, 1))
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form.get('username')
        pwd = request.form.get('password')
        if not uname or not pwd:
            return 'Username and password required', 400
        conn = get_db()
        cur = conn.cursor()
        try:
            hashed = hash_password(pwd)
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (uname, hashed))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return 'Username already exists', 400
    return render_template_string('''
        <h2>Register</h2>
        <form method="post">
          Username: <input name="username" required><br>
          Password: <input type="password" name="password" required><br>
          <input type="submit" value="Register">
        </form>
        <p>Already registered? <a href="{{ url_for('login') }}">Login</a></p>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form.get('username')
        pwd = request.form.get('password')
        if not uname or not pwd:
            return 'Username and password required', 400
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (uname,))
        row = cur.fetchone()
        conn.close()
        if row and hash_password(pwd) == row[0]:
            session['username'] = uname
            return redirect(url_for('welcome'))
        else:
            return 'Invalid username or password', 401
    return render_template_string('''
        <h2>Login</h2>
        <form method="post">
          Username: <input name="username" required><br>
          Password: <input type="password" name="password" required><br>
          <input type="submit" value="Login">
        </form>
        <p>New user? <a href="{{ url_for('register') }}">Register here</a></p>
    ''')

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Welcome {{ user }}</h2>
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
        {% if is_admin %}
        <p><a href="{{ url_for('admin') }}">Admin area</a></p>
        {% endif %}
    ''', user=session['username'], is_admin=is_admin(session['username']))

def is_admin(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT is_admin FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row and row[0] == 1

@app.route('/admin')
def admin():
    if 'username' not in session or not is_admin(session['username']):
        return 'Unauthorized', 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users ORDER BY id")
    users = cur.fetchall()
    conn.close()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <table border="1">
          <tr><th>ID</th><th>Username</th></tr>
          {% for id, username in users %}
          <tr><td>{{ id }}</td><td>{{ username }}</td></tr>
          {% endfor %}
        </table>
        <p><a href="{{ url_for('welcome') }}">Back to welcome</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)