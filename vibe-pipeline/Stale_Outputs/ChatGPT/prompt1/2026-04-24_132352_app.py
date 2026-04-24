from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

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
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        # Create a default admin user if none exists
        c.execute("SELECT * FROM users WHERE is_admin=1")
        if not c.fetchone():
            admin_pass = hashlib.sha256("adminpass".encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      ("admin", admin_pass, 1))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return c.fetchone()

def add_user(username, password):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        hashed = hashlib.sha256(password.encode()).hexdigest()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username and password:
            if add_user(username, password):
                return redirect(url_for('login'))
            else:
                error = "Username already exists."
        else:
            error = "Please enter both username and password."
        return render_template_string(REGISTRATION_PAGE, error=error)
    return render_template_string(REGISTRATION_PAGE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = get_user(username)
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if user and user[2] == hashed:
            session['user'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        error = "Invalid username or password."
        return render_template_string(LOGIN_PAGE, error=error)
    return render_template_string(LOGIN_PAGE)

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_PAGE, username=session['user'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'user' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT username, is_admin FROM users")
        users = c.fetchall()
    return render_template_string(ADMIN_PAGE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

REGISTRATION_PAGE = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  Username: <input type=text name=username><br>
  Password: <input type=password name=password><br>
  <input type=submit value=Register>
</form>
<a href="{{ url_for('login') }}">Login</a>
'''

LOGIN_PAGE = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  Username: <input type=text name=username><br>
  Password: <input type=password name=password><br>
  <input type=submit value=Login>
</form>
<a href="{{ url_for('register') }}">Register</a>
'''

WELCOME_PAGE = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
  <p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_PAGE = '''
<!doctype html>
<title>Admin Area</title>
<h2>All Registered Users</h2>
<table border=1>
  <tr><th>Username</th><th>Is Admin</th></tr>
  {% for user, admin in users %}
  <tr><td>{{ user }}</td><td>{{ 'Yes' if admin else 'No' }}</td></tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)