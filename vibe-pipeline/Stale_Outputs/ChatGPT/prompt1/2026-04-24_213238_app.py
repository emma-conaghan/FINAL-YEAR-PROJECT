from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      admin INTEGER DEFAULT 0)''')
        # Create default admin if not exists
        c.execute("SELECT * FROM users WHERE username = ?", ('admin',))
        if not c.fetchone():
            pwd_hash = hashlib.sha256(b'admin').hexdigest()
            c.execute("INSERT INTO users (username, password, admin) VALUES (?, ?, 1)", ('admin', pwd_hash))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, admin FROM users WHERE username = ?", (username,))
        return c.fetchone()

def add_user(username, password):
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, pwd_hash))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

@app.route('/')
def index():
    if "user_id" in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username and password:
            if add_user(username, password):
                return redirect(url_for('login'))
            else:
                error = "Username already exists."
        else:
            error = "Please provide username and password."
        return render_template_string(REG_TEMPLATE, error=error)
    return render_template_string(REG_TEMPLATE, error=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user:
            pwd_hash = hashlib.sha256(password.encode()).hexdigest()
            if pwd_hash == user[2]:
                session['user_id'] = user[0]
                session['username'] = user[1]
                session['admin'] = user[3]
                return redirect(url_for('welcome'))
        error = "Invalid username or password."
        return render_template_string(LOGIN_TEMPLATE, error=error)
    return render_template_string(LOGIN_TEMPLATE, error=None)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], admin=session['admin'])

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('admin'):
        return redirect(url_for('login'))
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT username, admin FROM users")
        users = c.fetchall()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

REG_TEMPLATE = '''
<!doctype html>
<title>Register</title>
<h1>Register</h1>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  Username:<br><input type=text name=username required><br>
  Password:<br><input type=password name=password required><br><br>
  <input type=submit value=Register>
</form>
<a href="{{ url_for('login') }}">Login</a>
'''

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h1>Login</h1>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  Username:<br><input type=text name=username required><br>
  Password:<br><input type=password name=password required><br><br>
  <input type=submit value=Login>
</form>
<a href="{{ url_for('register') }}">Register</a>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h1>Welcome {{ username }}!</h1>
{% if admin %}
<p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin Area</title>
<h1>Registered Users</h1>
<table border=1>
<tr><th>Username</th><th>Admin</th></tr>
{% for user, admin in users %}
<tr><td>{{ user }}</td><td>{{ 'Yes' if admin else 'No' }}</td></tr>
{% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)