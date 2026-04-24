from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0)''')
        # Create a default admin user if not exists
        cur.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            hashed = hashlib.sha256('adminpass'.encode()).hexdigest()
            cur.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ('admin', hashed))
        con.commit()

def get_user(username):
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return cur.fetchone()

def add_user(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            con.commit()
            return True
        except sqlite3.IntegrityError:
            return False

def check_password(stored_hash, password):
    return stored_hash == hashlib.sha256(password.encode()).hexdigest()

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
        if username == '' or password == '':
            error = "Username and password cannot be empty."
            return render_template_string(REG_TEMP, error=error)
        if add_user(username, password):
            return redirect(url_for('login'))
        else:
            error = "Username already exists."
            return render_template_string(REG_TEMP, error=error)
    return render_template_string(REG_TEMP, error=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = get_user(username)
        if user and check_password(user[2], password):
            session['user'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        else:
            error = "Invalid username or password."
            return render_template_string(LOGIN_TEMP, error=error)
    return render_template_string(LOGIN_TEMP, error=None)

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMP, user=session['user'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'user' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("SELECT username, is_admin FROM users")
        users = cur.fetchall()
    return render_template_string(ADMIN_TEMP, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

LOGIN_TEMP = """
<!doctype html>
<title>Login</title>
<h1>Login</h1>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  Username: <input type=text name=username>
  <br>
  Password: <input type=password name=password>
  <br>
  <input type=submit value=Login>
</form>
<p>New user? <a href="{{ url_for('register') }}">Register here</a></p>
"""

REG_TEMP = """
<!doctype html>
<title>Register</title>
<h1>Register</h1>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  Username: <input type=text name=username>
  <br>
  Password: <input type=password name=password>
  <br>
  <input type=submit value=Register>
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
"""

WELCOME_TEMP = """
<!doctype html>
<title>Welcome</title>
<h1>Welcome, {{ user }}!</h1>
{% if is_admin %}
  <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
"""

ADMIN_TEMP = """
<!doctype html>
<title>Admin Area</title>
<h1>All Registered Users</h1>
<table border=1>
  <tr><th>Username</th><th>Admin</th></tr>
  {% for user, is_admin in users %}
    <tr><td>{{ user }}</td><td>{{ 'Yes' if is_admin else 'No' }}</td></tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
"""

if __name__ == '__main__':
    init_db()
    app.run(debug=True)