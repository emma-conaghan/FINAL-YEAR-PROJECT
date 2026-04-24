from flask import Flask, request, redirect, url_for, session, render_template_string
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as con:
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER NOT NULL DEFAULT 0)''')
        # Create a default admin user if not exists
        cur.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            cur.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                        ('admin', generate_password_hash('admin123'), 1))
        con.commit()

def get_user(username):
    con = sqlite3.connect(DATABASE)
    cur = con.cursor()
    cur.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    con.close()
    return user

def get_all_users():
    con = sqlite3.connect(DATABASE)
    cur = con.cursor()
    cur.execute("SELECT username, is_admin FROM users")
    users = cur.fetchall()
    con.close()
    return users

def add_user(username, password):
    try:
        con = sqlite3.connect(DATABASE)
        cur = con.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, generate_password_hash(password)))
        con.commit()
        con.close()
        return True
    except sqlite3.IntegrityError:
        return False

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
            return render_template_string(REG_TEMPLATE, error="Please fill out all fields")
        if add_user(username, password):
            return redirect(url_for('login'))
        else:
            return render_template_string(REG_TEMPLATE, error="Username already exists")
    return render_template_string(REG_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = user[1]
            session['is_admin'] = user[3]
            return redirect(url_for('welcome'))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password")
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    return render_template_string(WELCOME_TEMPLATE, username=username, is_admin=session.get('is_admin'))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    users = get_all_users()
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
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Register>
</form>
{% if error %}
<p style="color:red;">{{ error }}</p>
{% endif %}
<p>Already have an account? <a href="/login">Login here</a>.</p>
'''

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Login>
</form>
{% if error %}
<p style="color:red;">{{ error }}</p>
{% endif %}
<p>Don't have an account? <a href="/register">Register here</a>.</p>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="/admin">Go to admin area</a></p>
{% endif %}
<p><a href="/logout">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin Area</title>
<h2>Registered Users</h2>
<table border=1>
  <tr><th>Username</th><th>Admin</th></tr>
  {% for user, is_admin in users %}
  <tr><td>{{ user }}</td><td>{{ 'Yes' if is_admin else 'No' }}</td></tr>
  {% endfor %}
</table>
<p><a href="/welcome">Back to welcome page</a></p>
<p><a href="/logout">Logout</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)