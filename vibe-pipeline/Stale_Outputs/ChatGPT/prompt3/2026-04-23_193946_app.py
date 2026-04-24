from flask import Flask, render_template_string, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
        ''')
        # Create default admin if not exists
        c.execute("SELECT * FROM users WHERE username = ?", ('admin',))
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      ('admin', generate_password_hash('admin'), 1))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return c.fetchone()

def create_user(username, password):
    try:
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                      (username, generate_password_hash(password)))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password'
            return render_template_string(LOGIN_HTML, error=error)
    return render_template_string(LOGIN_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if create_user(username, password):
            return redirect(url_for('login'))
        else:
            error = 'Username already exists'
            return render_template_string(REGISTER_HTML, error=error)
    return render_template_string(REGISTER_HTML)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_HTML, username=session['username'], is_admin=session['is_admin'])

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, is_admin FROM users")
        users = c.fetchall()
    return render_template_string(ADMIN_HTML, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

LOGIN_HTML = '''
<!doctype html>
<title>Login</title>
<h1>Login</h1>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Login>
</form>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
'''

REGISTER_HTML = '''
<!doctype html>
<title>Register</title>
<h1>Register</h1>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Register>
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
'''

WELCOME_HTML = '''
<!doctype html>
<title>Welcome</title>
<h1>Welcome, {{ username }}!</h1>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_HTML = '''
<!doctype html>
<title>Admin Area</title>
<h1>Admin Area - Registered Users</h1>
<table border=1 cellpadding=5>
<tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
{% for id, username, is_admin in users %}
<tr>
  <td>{{ id }}</td>
  <td>{{ username }}</td>
  <td>{{ 'Yes' if is_admin else 'No' }}</td>
</tr>
{% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)