from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DB_NAME = 'users.db'

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )''')
        conn.commit()
        # Create default admin if not exists
        c.execute("SELECT * FROM users WHERE username = ?", ('admin',))
        if not c.fetchone():
            hashed_pw = hashlib.sha256('admin'.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', hashed_pw, 1))
            conn.commit()

init_db()

def query_db(query, args=(), one=False):
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute(query, args)
        rv = cur.fetchall()
        conn.commit()
        return (rv[0] if rv else None) if one else rv

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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
            return render_template_string(REG_TEMPLATE, error="Please fill all fields")
        existing = query_db('SELECT * FROM users WHERE username = ?', (username,), one=True)
        if existing:
            return render_template_string(REG_TEMPLATE, error="Username already taken")
        hashed_pw = hash_password(password)
        try:
            query_db('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
        except:
            return render_template_string(REG_TEMPLATE, error="Registration failed")
        return redirect(url_for('login'))
    return render_template_string(REG_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return render_template_string(LOGIN_TEMPLATE, error="Please fill all fields")
        user = query_db('SELECT * FROM users WHERE username = ?', (username,), one=True)
        if not user:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password")
        hashed_pw = hash_password(password)
        if hashed_pw != user[2]:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password")
        session['username'] = username
        session['is_admin'] = bool(user[3])
        return redirect(url_for('welcome'))
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    users = query_db('SELECT username, is_admin FROM users ORDER BY id')
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
  <input type=text name=username placeholder="Username" required>
  <input type=password name=password placeholder="Password" required>
  <input type=submit value=Register>
</form>
<p style="color:red;">{{ error }}</p>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
'''

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  <input type=text name=username placeholder="Username" required>
  <input type=password name=password placeholder="Password" required>
  <input type=submit value=Login>
</form>
<p style="color:red;">{{ error }}</p>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin - Users</title>
<h2>Registered Users</h2>
<table border=1 cellpadding=5 cellspacing=0>
  <tr><th>Username</th><th>Is Admin</th></tr>
  {% for user, is_admin in users %}
  <tr><td>{{ user }}</td><td>{{ 'Yes' if is_admin else 'No' }}</td></tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

if __name__ == '__main__':
    app.run(debug=True)