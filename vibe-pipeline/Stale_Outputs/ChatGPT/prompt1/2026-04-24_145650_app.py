from flask import Flask, request, redirect, render_template_string, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'secretkey123'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )''')
        # Create admin user if not exists
        admin = db.execute("SELECT * FROM users WHERE username = ?","admin").fetchone()
        if not admin:
            pw_hash = hashlib.sha256("adminpass".encode()).hexdigest()
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ("admin", pw_hash))
        db.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Username and password required'
        else:
            pw_hash = hash_password(password)
            try:
                with get_db() as db:
                    db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, pw_hash))
                    db.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = 'Username already taken'
    return render_template_string('''
    <h2>Register</h2>
    <form method="post">
      Username: <input type="text" name="username" required><br>
      Password: <input type="password" name="password" required><br>
      <input type="submit" value="Register">
    </form>
    {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
    <a href="{{ url_for('login') }}">Login</a>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        if not username or not password:
            error = "Username and password required"
        else:
            with get_db() as db:
                user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
                if user and user['password'] == hash_password(password):
                    session['username'] = username
                    session['is_admin'] = bool(user['is_admin'])
                    return redirect(url_for('welcome'))
                else:
                    error = "Invalid credentials"
    return render_template_string('''
    <h2>Login</h2>
    <form method="post">
      Username: <input type="text" name="username" required><br>
      Password: <input type="password" name="password" required><br>
      <input type="submit" value="Login">
    </form>
    {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
    <a href="{{ url_for('register') }}">Register</a>
    ''', error=error)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
    <h2>Welcome {{ session.username }}!</h2>
    {% if session.is_admin %}
    <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
    {% endif %}
    <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''')

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    with get_db() as db:
        users = db.execute("SELECT username, is_admin FROM users ORDER BY username").fetchall()
    return render_template_string('''
    <h2>Admin Area - Registered Users</h2>
    <table border=1>
      <tr><th>Username</th><th>Is Admin</th></tr>
      {% for user in users %}
      <tr><td>{{ user['username'] }}</td><td>{{ 'Yes' if user['is_admin'] else 'No' }}</td></tr>
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