from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'replace_with_a_secret_key'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  is_admin INTEGER DEFAULT 0)''')
    # Create default admin if not exists
    c.execute("SELECT * FROM users WHERE username = ?", ('admin',))
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)",
                  ('admin', generate_password_hash('admin')))
    conn.commit()
    conn.close()

init_db()

login_template = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
{% if error %}<p style="color: red;">{{ error }}</p>{% endif %}
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Login>
</form>
<p>New? <a href="{{ url_for('register') }}">Register here</a></p>
'''

register_template = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
{% if error %}<p style="color: red;">{{ error }}</p>{% endif %}
{% if success %}<p style="color: green;">{{ success }}</p>{% endif %}
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Register>
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
'''

welcome_template = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ user }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Page</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

admin_template = '''
<!doctype html>
<title>Admin Area</title>
<h2>Registered Users</h2>
<table border=1>
  <tr><th>ID</th><th>Username</th><th>Admin</th></tr>
  {% for user in users %}
  <tr>
    <td>{{ user['id'] }}</td>
    <td>{{ user['username'] }}</td>
    <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
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
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid credentials'
    return render_template_string(login_template, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if len(username) < 3 or len(password) < 3:
            error = 'Username and password must be at least 3 characters long.'
        else:
            hashed = generate_password_hash(password)
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password) VALUES (?,?)", (username, hashed))
                conn.commit()
                conn.close()
                success = 'Registration successful. You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already taken.'
    return render_template_string(register_template, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_template, user=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, username, is_admin FROM users")
    users = c.fetchall()
    conn.close()
    return render_template_string(admin_template, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)