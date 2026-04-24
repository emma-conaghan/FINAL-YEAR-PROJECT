from flask import Flask, request, redirect, url_for, render_template_string, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

def init_db():
    with sqlite3.connect('app.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      is_admin INTEGER DEFAULT 0)''')
        # Add default admin user
        c.execute("SELECT * FROM users WHERE username = 'admin'")
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      ('admin', generate_password_hash('admin123'), 1))
        conn.commit()

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
            return render_template_string(REG_TEMPLATE, error="Username and password required.")
        with sqlite3.connect('app.db') as conn:
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                          (username, generate_password_hash(password)))
                conn.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                return render_template_string(REG_TEMPLATE, error="Username already taken.")
    return render_template_string(REG_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        with sqlite3.connect('app.db') as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            row = c.fetchone()
            if row and check_password_hash(row[0], password):
                session['username'] = username
                return redirect(url_for('welcome'))
        return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials.")
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'])

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    with sqlite3.connect('app.db') as conn:
        c = conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE username = ?", (session['username'],))
        row = c.fetchone()
        if not row or row[0] == 0:
            return "Access denied", 403
        c.execute("SELECT id, username, is_admin FROM users")
        users = c.fetchall()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

REG_TEMPLATE = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method="post">
  Username: <input type="text" name="username" autofocus><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Register">
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a>.</p>
'''

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method="post">
  Username: <input type="text" name="username" autofocus><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Login">
</form>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a>.</p>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
<p><a href="{{ url_for('admin') }}">Admin Area</a> (admin only)</p>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin - All Users</title>
<h2>All Registered Users</h2>
<table border=1>
  <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
  {% for u in users %}
  <tr>
    <td>{{ u[0] }}</td>
    <td>{{ u[1] }}</td>
    <td>{{ "Yes" if u[2] else "No" }}</td>
  </tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)