from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0
        )
        ''')
    # Create default admin user
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            pw_hash = generate_password_hash('admin')
            conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ('admin', pw_hash))

init_db()

def get_user(username):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return cur.fetchone()

def get_all_users():
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.execute("SELECT username, is_admin FROM users")
        return cur.fetchall()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            return render_template_string(REG_TEMPLATE, error="Username and password required")
        if get_user(username):
            return render_template_string(REG_TEMPLATE, error="Username already exists")
        pw_hash = generate_password_hash(password)
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, pw_hash))
        return redirect(url_for('login'))
    return render_template_string(REG_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = user[1]
            session['is_admin'] = user[3]
            return redirect(url_for('welcome'))
        return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials")
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], is_admin=session.get('is_admin', 0))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    users = get_all_users()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method="POST">
  Username: <input type="text" name="username" required><br><br>
  Password: <input type="password" name="password" required><br><br>
  <button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
'''

REG_TEMPLATE = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method="POST">
  Username: <input type="text" name="username" required><br><br>
  Password: <input type="password" name="password" required><br><br>
  <button type="submit">Register</button>
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin Area</title>
<h2>All Registered Users</h2>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Username</th><th>Admin</th></tr>
{% for user, is_admin in users %}
<tr><td>{{ user }}</td><td>{{ 'Yes' if is_admin else 'No' }}</td></tr>
{% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

if __name__ == '__main__':
    app.run(debug=True)