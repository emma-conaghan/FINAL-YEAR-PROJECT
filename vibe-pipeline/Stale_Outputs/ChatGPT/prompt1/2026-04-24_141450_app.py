from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secret-key'

def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    is_admin INTEGER DEFAULT 0
                  )''')
    db.commit()
    cursor = db.execute("SELECT * FROM users WHERE username = 'admin'")
    if cursor.fetchone() is None:
        hashed = generate_password_hash('admin')
        db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ('admin', hashed))
        db.commit()
    db.close()

init_db()

LOGIN_HTML = '''
<!doctype html>
<title>Login</title>
<h1>Login</h1>
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Login>
</form>
<a href="{{ url_for('register') }}">Register</a>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
'''

REGISTER_HTML = '''
<!doctype html>
<title>Register</title>
<h1>Register</h1>
<form method=post>
  Username: <input type=text name=username required><br>
  Password: <input type=password name=password required><br>
  <input type=submit value=Register>
</form>
<a href="{{ url_for('login') }}">Login</a>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
{% if success %}<p style="color:green;">{{ success }}</p>{% endif %}
'''

WELCOME_HTML = '''
<!doctype html>
<title>Welcome</title>
<h1>Welcome {{ username }}!</h1>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_HTML = '''
<!doctype html>
<title>Admin Area</title>
<h1>Registered Users</h1>
<table border=1>
<tr><th>ID</th><th>Username</th><th>Admin</th></tr>
{% for user in users %}
<tr>
  <td>{{ user.id }}</td>
  <td>{{ user.username }}</td>
  <td>{{ 'Yes' if user.is_admin else 'No' }}</td>
</tr>
{% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        db.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password'
    return render_template_string(LOGIN_HTML, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if len(username) < 3 or len(password) < 3:
            error = 'Username and password must be at least 3 characters long'
        else:
            hashed = generate_password_hash(password)
            try:
                db = get_db()
                db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
                db.commit()
                db.close()
                success = 'Registration successful. You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already taken'
    return render_template_string(REGISTER_HTML, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_HTML, username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    db = get_db()
    users = db.execute('SELECT id, username, is_admin FROM users ORDER BY id').fetchall()
    db.close()
    return render_template_string(ADMIN_HTML, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)