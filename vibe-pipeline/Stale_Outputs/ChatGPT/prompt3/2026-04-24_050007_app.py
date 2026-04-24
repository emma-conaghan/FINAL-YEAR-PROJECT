from flask import Flask, request, redirect, render_template_string, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DB_NAME = 'users.db'

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, is_admin INTEGER DEFAULT 0)")
        db.commit()
    # Create admin user if not exists
    with get_db() as db:
        cur = db.execute("SELECT * FROM users WHERE username = ?", ('admin',))
        if cur.fetchone() is None:
            hashed = generate_password_hash('adminpass')
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', hashed, 1))
            db.commit()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            return render_template_string(REG_FORM, error='Username and password required')
        with get_db() as db:
            try:
                hashed = generate_password_hash(password)
                db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                db.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                return render_template_string(REG_FORM, error='Username already exists')
    return render_template_string(REG_FORM, error='')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        with get_db() as db:
            cur = db.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cur.fetchone()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = bool(user['is_admin'])
                return redirect(url_for('welcome'))
        return render_template_string(LOGIN_FORM, error='Invalid username or password')
    return render_template_string(LOGIN_FORM, error='')

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_PAGE, username=session['username'], is_admin=session['is_admin'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with get_db() as db:
        cur = db.execute("SELECT username, is_admin FROM users ORDER BY username")
        users = cur.fetchall()
    return render_template_string(ADMIN_PAGE, users=users)

REG_FORM = '''
<!doctype html>
<title>Register</title>
<h1>Register</h1>
<form method=post>
  <label>Username: <input type=text name=username></label><br>
  <label>Password: <input type=password name=password></label><br>
  <input type=submit value=Register>
</form>
<p style="color:red;">{{ error }}</p>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a>.</p>
'''

LOGIN_FORM = '''
<!doctype html>
<title>Login</title>
<h1>Login</h1>
<form method=post>
  <label>Username: <input type=text name=username></label><br>
  <label>Password: <input type=password name=password></label><br>
  <input type=submit value=Login>
</form>
<p style="color:red;">{{ error }}</p>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a>.</p>
'''

WELCOME_PAGE = '''
<!doctype html>
<title>Welcome</title>
<h1>Welcome, {{ username }}!</h1>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_PAGE = '''
<!doctype html>
<title>Admin Area</title>
<h1>Admin Area - Registered Users</h1>
<table border=1>
  <tr><th>Username</th><th>Admin</th></tr>
  {% for user in users %}
  <tr>
    <td>{{ user['username'] }}</td>
    <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
  </tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)