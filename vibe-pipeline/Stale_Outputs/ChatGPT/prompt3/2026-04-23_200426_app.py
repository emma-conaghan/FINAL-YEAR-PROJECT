from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'change_this_secret'

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )''')
        # Create a default admin if doesn't exist
        c.execute("SELECT id FROM users WHERE username = 'admin'")
        if not c.fetchone():
            pw_hash = hashlib.sha256('admin'.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ('admin', pw_hash))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return c.fetchone()

def add_user(username, password):
    try:
        with sqlite3.connect(DB) as conn:
            c = conn.cursor()
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, pw_hash))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False

def all_users():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT username, is_admin FROM users")
        return c.fetchall()

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
            return render_template_string(REG_TEMPLATE, error="Username and password required.")
        if add_user(username, password):
            return redirect(url_for('login'))
        else:
            return render_template_string(REG_TEMPLATE, error="Username already taken.")
    return render_template_string(REG_TEMPLATE, error='')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user:
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            if pw_hash == user[2]:
                session['username'] = user[1]
                session['is_admin'] = bool(user[3])
                return redirect(url_for('welcome'))
        return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password.")
    return render_template_string(LOGIN_TEMPLATE, error='')

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    users = all_users()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

REG_TEMPLATE = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
<form method=post>
  <input placeholder="Username" name=username required>
  <input type=password placeholder="Password" name=password required>
  <button type=submit>Register</button>
</form>
<p style="color:red;">{{ error }}</p>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
'''

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  <input placeholder="Username" name=username required>
  <input type=password placeholder="Password" name=password required>
  <button type=submit>Login</button>
</form>
<p style="color:red;">{{ error }}</p>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Admin area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin Area</title>
<h2>Registered Users</h2>
<table border=1>
  <tr><th>Username</th><th>Is Admin</th></tr>
  {% for user, isadmin in users %}
  <tr><td>{{ user }}</td><td>{{ "Yes" if isadmin else "No" }}</td></tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)