from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'a_very_secret_key_for_sessions'

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER NOT NULL DEFAULT 0
                    )''')
        conn.commit()
    # Create default admin if not exists
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = 'admin'")
        if not c.fetchone():
            hashed = hashlib.sha256('admin'.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', hashed, 1))
            conn.commit()

def get_user(username):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
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

def check_password(password, hashed):
    return hashlib.sha256(password.encode()).hexdigest() == hashed

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Username and password required.'
        elif get_user(username):
            error = 'Username already taken.'
        else:
            if add_user(username, password):
                return redirect(url_for('login'))
            else:
                error = 'Error registering user.'
    return render_template_string('''
    <h2>Register</h2>
    <form method="post">
        <input placeholder="Username" name="username" required>
        <input type="password" placeholder="Password" name="password" required>
        <button type="submit">Register</button>
    </form>
    <p style="color:red;">{{error}}</p>
    <p>Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user is None or not check_password(password, user[2]):
            error = 'Invalid username or password.'
        else:
            session['username'] = username
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
    return render_template_string('''
    <h2>Login</h2>
    <form method="post">
        <input placeholder="Username" name="username" required>
        <input type="password" placeholder="Password" name="password" required>
        <button type="submit">Login</button>
    </form>
    <p style="color:red;">{{error}}</p>
    <p>No account? <a href="{{ url_for('register') }}">Register</a></p>
    ''', error=error)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
    <h2>Welcome {{username}}!</h2>
    {% if is_admin %}
    <p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
    {% endif %}
    <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if not session.get('is_admin', False):
        return redirect(url_for('login'))
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT username, is_admin FROM users ORDER BY username")
        users = c.fetchall()
    return render_template_string('''
    <h2>Admin Area - Registered Users</h2>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr><th>Username</th><th>Admin</th></tr>
        {% for user, admin in users %}
        <tr><td>{{ user }}</td><td>{{ 'Yes' if admin else 'No' }}</td></tr>
        {% endfor %}
    </table>
    <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)