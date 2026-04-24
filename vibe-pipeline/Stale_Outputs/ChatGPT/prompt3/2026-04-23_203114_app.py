from flask import Flask, request, redirect, url_for, render_template_string, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change in production

DB = 'users.db'


def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0
            )
        ''')
        conn.commit()
        # Create default admin if not exists
        c.execute("SELECT * FROM users WHERE username='admin'")
        if not c.fetchone():
            hashed = hashlib.sha256('adminpass'.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      ('admin', hashed, 1))
            conn.commit()


def get_user(username):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_admin FROM users WHERE username=?", (username,))
        return c.fetchone()


def add_user(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        with sqlite3.connect(DB) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username == '' or password == '':
            return render_template_string(REG_HTML, error="Username and password required")
        if add_user(username, password):
            return redirect(url_for('login'))
        else:
            return render_template_string(REG_HTML, error="Username already taken")
    return render_template_string(REG_HTML, error='')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user:
            hashed = hashlib.sha256(password.encode()).hexdigest()
            if hashed == user[2]:
                session['user'] = user[1]
                session['is_admin'] = bool(user[3])
                return redirect(url_for('welcome'))
        return render_template_string(LOGIN_HTML, error="Invalid username or password")
    return render_template_string(LOGIN_HTML, error='')


@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_HTML, user=session['user'], is_admin=session.get('is_admin', False))


@app.route('/admin')
def admin():
    if 'user' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT username, is_admin FROM users")
        users = c.fetchall()
    return render_template_string(ADMIN_HTML, users=users)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


REG_HTML = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
<form method=post>
  <input type=text name=username placeholder="Username" required>
  <input type=password name=password placeholder="Password" required>
  <input type=submit value=Register>
</form>
<a href="{{ url_for('login') }}">Login</a>
'''

LOGIN_HTML = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
<form method=post>
  <input type=text name=username placeholder="Username" required>
  <input type=password name=password placeholder="Password" required>
  <input type=submit value=Login>
</form>
<a href="{{ url_for('register') }}">Register</a>
'''

WELCOME_HTML = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome {{ user }}</h2>
{% if is_admin %}
  <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_HTML = '''
<!doctype html>
<title>Admin Area</title>
<h2>Registered Users</h2>
<table border=1 cellpadding=5 cellspacing=0>
  <tr><th>Username</th><th>Is Admin</th></tr>
  {% for username, is_admin in users %}
    <tr><td>{{ username }}</td><td>{{ 'Yes' if is_admin else 'No' }}</td></tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)