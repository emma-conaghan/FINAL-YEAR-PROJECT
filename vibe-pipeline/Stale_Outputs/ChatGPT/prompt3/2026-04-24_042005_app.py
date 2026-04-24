from flask import Flask, request, redirect, url_for, render_template_string, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'

DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as con:
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users
                       (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0)''')
        # Add default admin if not exists
        cur.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if not cur.fetchone():
            hashed = hashlib.sha256('admin'.encode()).hexdigest()
            cur.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                        ('admin', hashed, 1))
        con.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as con:
        cur = con.cursor()
        cur.execute('SELECT id, username, password, is_admin FROM users WHERE username = ?', (username,))
        return cur.fetchone()

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
            return render_template_string(register_template, error="Enter username and password.")
        if get_user(username):
            return render_template_string(register_template, error="Username already exists.")
        hashed = hashlib.sha256(password.encode()).hexdigest()
        with sqlite3.connect(DATABASE) as con:
            cur = con.cursor()
            try:
                cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
                con.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                return render_template_string(register_template, error="Username already exists.")
    return render_template_string(register_template, error='')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if user and user[2] == hashed:
            session['username'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        return render_template_string(login_template, error="Invalid username or password.")
    return render_template_string(login_template, error='')

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_template, username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with sqlite3.connect(DATABASE) as con:
        cur = con.cursor()
        cur.execute('SELECT username, is_admin FROM users ORDER BY username')
        users = cur.fetchall()
    return render_template_string(admin_template, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

register_template = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Register>
</form>
<p style="color:red;">{{ error }}</p>
<p>Already registered? <a href="{{ url_for('login') }}">Login here</a></p>
'''

login_template = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Login>
</form>
<p style="color:red;">{{ error }}</p>
<p>New user? <a href="{{ url_for('register') }}">Register here</a></p>
'''

welcome_template = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

admin_template = '''
<!doctype html>
<title>Admin Area</title>
<h2>Admin Area - Registered Users</h2>
<table border=1>
  <tr><th>Username</th><th>Admin</th></tr>
  {% for user, admin in users %}
    <tr><td>{{ user }}</td><td>{{ 'Yes' if admin else 'No' }}</td></tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)