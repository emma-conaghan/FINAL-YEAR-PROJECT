from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'secret_for_session'

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, is_admin INTEGER DEFAULT 0)')
        # Create default admin if not exists
        c.execute("SELECT * FROM users WHERE username='admin'")
        if not c.fetchone():
            hashed = hashlib.sha256('adminpass'.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', hashed, 1))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, password, is_admin FROM users WHERE username=?', (username,))
        return c.fetchone()

def add_user(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        with sqlite3.connect(DB) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        if username and password:
            if add_user(username, password):
                return redirect(url_for('login'))
            else:
                error = "Username already exists."
        else:
            error = "Please provide username and password."
        return render_template_string(REG_TEMPLATE, error=error)
    return render_template_string(REG_TEMPLATE, error='')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        user = get_user(username)
        if user:
            _, _user, pw_hash, is_admin = user
            if hashlib.sha256(password.encode()).hexdigest() == pw_hash:
                session['user_id'] = user[0]
                session['username'] = _user
                session['is_admin'] = is_admin
                return redirect(url_for('welcome'))
        error = "Invalid username or password."
        return render_template_string(LOGIN_TEMPLATE, error=error)
    return render_template_string(LOGIN_TEMPLATE, error='')

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], is_admin=session.get('is_admin',0))

@app.route('/admin')
def admin_area():
    if 'user_id' not in session or session.get('is_admin') != 1:
        return redirect(url_for('login'))
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('SELECT username, is_admin FROM users')
        users = c.fetchall()
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
  <input type=text name=username placeholder="Username" required><br>
  <input type=password name=password placeholder="Password" required><br>
  <input type=submit value=Register>
</form>
<p style="color:red;">{{ error }}</p>
<p>Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
'''

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  <input type=text name=username placeholder="Username" required><br>
  <input type=password name=password placeholder="Password" required><br>
  <input type=submit value=Login>
</form>
<p style="color:red;">{{ error }}</p>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register</a></p>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome {{ username }}!</h2>
{% if is_admin == 1 %}
<p><a href="{{ url_for('admin_area') }}">Go to Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin Area</title>
<h2>Admin Area - All Registered Users</h2>
<table border=1>
  <tr><th>Username</th><th>Is Admin</th></tr>
  {% for user, is_admin in users %}
  <tr><td>{{ user }}</td><td>{{ 'Yes' if is_admin == 1 else 'No' }}</td></tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)