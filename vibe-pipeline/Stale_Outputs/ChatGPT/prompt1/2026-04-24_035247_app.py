from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DB_NAME = 'users.db'

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0
                        )''')
        existing_admin = conn.execute("SELECT * FROM users WHERE is_admin=1").fetchone()
        if not existing_admin:
            # Create default admin user
            conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                         ('admin', generate_password_hash('adminpass'), 1))

init_db()

login_template = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  <input name=username placeholder="Username" required>
  <input name=password type=password placeholder="Password" required>
  <input type=submit value=Login>
</form>
<a href="{{ url_for('register') }}">Register</a>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
'''

register_template = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
<form method=post>
  <input name=username placeholder="Username" required>
  <input name=password type=password placeholder="Password" required>
  <input type=submit value=Register>
</form>
<a href="{{ url_for('login') }}">Login</a>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
'''

welcome_template = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
  <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

admin_template = '''
<!doctype html>
<title>Admin Area</title>
<h2>Registered Users</h2>
<table border=1>
  <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
  {% for user in users %}
    <tr>
      <td>{{ user[0] }}</td>
      <td>{{ user[1] }}</td>
      <td>{{ 'Yes' if user[3] else 'No' }}</td>
    </tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(DB_NAME) as conn:
            user = conn.execute("SELECT id, username, password, is_admin FROM users WHERE username=?", (username,)).fetchone()
            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                session['username'] = user[1]
                session['is_admin'] = bool(user[3])
                return redirect(url_for('welcome'))
            else:
                error = 'Invalid username or password'
    return render_template_string(login_template, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_pwd = generate_password_hash(password)
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pwd))
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = 'Username already taken'
    return render_template_string(register_template, error=error)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_template, username=session['username'], is_admin=session['is_admin'])

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with sqlite3.connect(DB_NAME) as conn:
        users = conn.execute("SELECT * FROM users").fetchall()
    return render_template_string(admin_template, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)