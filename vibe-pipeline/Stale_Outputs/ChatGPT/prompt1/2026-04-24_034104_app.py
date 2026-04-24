from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            is_admin INTEGER DEFAULT 0
        )
        ''')
        # Create default admin user if not exists (username: admin, password: admin)
        cur.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if not cur.fetchone():
            pwd = hashlib.sha256('admin'.encode()).hexdigest()
            cur.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', pwd, 1))
        con.commit()

def get_user(username):
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute('SELECT id, username, password, is_admin FROM users WHERE username = ?', (username,))
        return cur.fetchone()

def add_user(username, password):
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        with sqlite3.connect(DB) as con:
            cur = con.cursor()
            cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, pwd_hash))
            con.commit()
        return True
    except sqlite3.IntegrityError:
        return False

@app.route('/')
def index():
    if 'user' in session:
        username = session['user']
        return render_template_string('''
            <h2>Welcome, {{username}}!</h2>
            <p><a href="{{ url_for('logout') }}">Logout</a></p>
            {% if is_admin %}
            <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
            {% endif %}
        ''', username=username, is_admin=session.get('is_admin', False))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Please enter both username and password.'
        else:
            if add_user(username, password):
                return redirect(url_for('login'))
            else:
                error = 'Username already taken.'
    return render_template_string('''
        <h2>Register</h2>
        <form method="post">
            Username:<br><input name="username" required><br>
            Password:<br><input type="password" name="password" required><br><br>
            <input type="submit" value="Register">
        </form>
        <p style="color:red;">{{error}}</p>
        <p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user:
            id_, uname, pwd_hash, is_admin = user
            if pwd_hash == hashlib.sha256(password.encode()).hexdigest():
                session['user'] = uname
                session['is_admin'] = bool(is_admin)
                return redirect(url_for('index'))
        error = 'Invalid username or password.'
    return render_template_string('''
        <h2>Login</h2>
        <form method="post">
            Username:<br><input name="username" required><br>
            Password:<br><input type="password" name="password" required><br><br>
            <input type="submit" value="Login">
        </form>
        <p style="color:red;">{{error}}</p>
        <p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
    ''', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if 'user' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute('SELECT username, is_admin FROM users')
        users = cur.fetchall()
    return render_template_string('''
        <h2>Admin Area</h2>
        <p><a href="{{ url_for('index') }}">Back to Home</a></p>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><th>Username</th><th>Is Admin</th></tr>
            {% for u, adm in users %}
            <tr><td>{{ u }}</td><td>{{ 'Yes' if adm else 'No' }}</td></tr>
            {% endfor %}
        </table>
    ''', users=users)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)