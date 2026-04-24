from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Insert an admin user if not present (username: admin, password: admin)
    admin = db.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()
    if not admin:
        db.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                   ('admin', generate_password_hash('admin'), 1))
    db.commit()
    db.close()

init_db()

signup_template = '''
<!doctype html>
<title>Register</title>
<h2>Register</h2>
<form method=post>
  <input type=text name=username placeholder="Username" required>
  <input type=password name=password placeholder="Password" required>
  <input type=submit value=Register>
</form>
<a href="{{ url_for('login') }}">Already have an account? Login</a>
<p style="color:red;">{{ error }}</p>
'''

login_template = '''
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  <input type=text name=username placeholder="Username" required>
  <input type=password name=password placeholder="Password" required>
  <input type=submit value=Login>
</form>
<a href="{{ url_for('signup') }}">Don't have an account? Register</a>
<p style="color:red;">{{ error }}</p>
'''

welcome_template = '''
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Admin area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

admin_template = '''
<!doctype html>
<title>Admin Area</title>
<h2>Registered Users</h2>
<table border=1>
  <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
  {% for user in users %}
  <tr>
    <td>{{ user.id }}</td>
    <td>{{ user.username }}</td>
    <td>{{ 'Yes' if user.is_admin else 'No' }}</td>
  </tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to welcome</a></p>
'''

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = ''
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            error = 'Please provide username and password.'
        else:
            db = get_db()
            existing = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            if existing:
                error = 'Username already exists.'
            else:
                hashed = generate_password_hash(password)
                db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed))
                db.commit()
                return redirect(url_for('login'))
            db.close()
    return render_template_string(signup_template, error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        db.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin'] == 1
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_template_string(login_template, error=error)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_template, username=session['username'], is_admin=session['is_admin'])

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    db = get_db()
    users = db.execute('SELECT * FROM users').fetchall()
    db.close()
    return render_template_string(admin_template, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)