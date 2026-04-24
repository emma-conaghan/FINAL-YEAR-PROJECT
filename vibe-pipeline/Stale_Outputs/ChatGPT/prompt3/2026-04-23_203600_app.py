from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'change_this_secret_key'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        # Insert default admin user if not exists
        admin_user = db.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
        if not admin_user:
            pw = hashlib.sha256("admin".encode()).hexdigest()
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", 
                       ('admin', pw, 1))
        db.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return render_template_string(REG_TEMPLATE, error="Username and password required.")
        with get_db() as db:
            existing = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                return render_template_string(REG_TEMPLATE, error="Username already taken.")
            hashed_pw = hash_password(password)
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            db.commit()
        return redirect(url_for('login'))
    return render_template_string(REG_TEMPLATE)

@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return render_template_string(LOGIN_TEMPLATE, error="Enter both username and password.")
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if not user or hash_password(password) != user['password']:
                return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password.")
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    is_admin = session.get('is_admin', False)
    return render_template_string(WELCOME_TEMPLATE, username=username, is_admin=is_admin)

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with get_db() as db:
        users = db.execute("SELECT username, is_admin FROM users").fetchall()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

REG_TEMPLATE = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
{% if error %}<p style="color:red;">{{error}}</p>{% endif %}
<form method="post">
  Username: <input name="username" required><br>
  Password: <input type="password" name="password" required><br>
  <input type="submit" value="Register">
</form>
<a href="{{ url_for('login') }}">Login</a>
'''

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
{% if error %}<p style="color:red;">{{error}}</p>{% endif %}
<form method="post">
  Username: <input name="username" required><br>
  Password: <input type="password" name="password" required><br>
  <input type="submit" value="Login">
</form>
<a href="{{ url_for('register') }}">Register</a>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{username}}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin Area</title>
<h2>Registered Users</h2>
<table border=1>
  <tr><th>Username</th><th>Admin</th></tr>
  {% for user in users %}
  <tr>
    <td>{{user['username']}}</td>
    <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
  </tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)