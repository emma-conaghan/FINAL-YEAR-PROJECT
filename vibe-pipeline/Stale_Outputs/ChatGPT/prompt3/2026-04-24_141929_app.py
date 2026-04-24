from flask import Flask, render_template_string, request, redirect, url_for, session
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
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        # create default admin user if not exists
        cur = conn.execute("SELECT * FROM users WHERE username = ?", ('admin',))
        admin = cur.fetchone()
        if not admin:
            conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", 
                         ('admin', generate_password_hash('admin123'), 1))

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            error = 'Username and password are required.'
            return render_template_string(REG_TEMPLATE, error=error)
        
        hashed_password = generate_password_hash(password)

        try:
            with get_db() as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                             (username, hashed_password))
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = 'Username is already taken.'
            return render_template_string(REG_TEMPLATE, error=error)

    return render_template_string(REG_TEMPLATE)

@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        error = None

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('welcome'))

        return render_template_string(LOGIN_TEMPLATE, error=error)

    return render_template_string(LOGIN_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], is_admin=session['is_admin'])

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with get_db() as conn:
        users = conn.execute("SELECT username, is_admin FROM users").fetchall()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

REG_TEMPLATE = '''
<!doctype html>
<title>Register</title>
<h1>Register</h1>
{% if error %}
<p style="color: red;">{{ error }}</p>
{% endif %}
<form method="post">
    <label>Username:</label><br>
    <input name="username" required><br>
    <label>Password:</label><br>
    <input type="password" name="password" required><br><br>
    <button type="submit">Register</button>
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a>.</p>
'''

LOGIN_TEMPLATE = '''
<!doctype html>
<title>Login</title>
<h1>Login</h1>
{% if error %}
<p style="color: red;">{{ error }}</p>
{% endif %}
<form method="post">
    <label>Username:</label><br>
    <input name="username" required><br>
    <label>Password:</label><br>
    <input type="password" name="password" required><br><br>
    <button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a>.</p>
'''

WELCOME_TEMPLATE = '''
<!doctype html>
<title>Welcome</title>
<h1>Welcome, {{ username }}!</h1>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Panel</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

ADMIN_TEMPLATE = '''
<!doctype html>
<title>Admin Panel</title>
<h1>Admin Panel - Registered Users</h1>
<table border="1">
    <tr>
        <th>Username</th>
        <th>Is Admin</th>
    </tr>
    {% for user in users %}
    <tr>
        <td>{{ user['username'] }}</td>
        <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
    </tr>
    {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome Page</a></p>
'''

if __name__ == '__main__':
    init_db()
    app.run(debug=True)