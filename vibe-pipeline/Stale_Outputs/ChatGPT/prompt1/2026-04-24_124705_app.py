from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'change_this_secret_key'

def init_db():
    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0
            )
        ''')
        # Create default admin user
        c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if not c.fetchone():
            admin_pass = hashlib.sha256('admin123'.encode()).hexdigest()
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', admin_pass, 1))
        conn.commit()

def get_user(username):
    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, password, is_admin FROM users WHERE username = ?', (username,))
        return c.fetchone()

def add_user(username, password_hash):
    try:
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password_hash))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('welcome'))
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Username and password are required'
        else:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if add_user(username, password_hash):
                return redirect(url_for('login'))
            else:
                error = 'Username already taken'
    return render_template_string('''
        <h2>Register</h2>
        <form method="POST">
            <label>Username: <input name="username"></label><br>
            <label>Password: <input type="password" name="password"></label><br>
            <button type="submit">Register</button>
        </form>
        <p style="color:red;">{{error}}</p>
        <p>Already have an account? <a href="{{url_for('login')}}">Login here</a></p>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('welcome'))
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user:
            _, uname, pw_hash, is_admin = user
            if hashlib.sha256(password.encode()).hexdigest() == pw_hash:
                session['user'] = uname
                session['is_admin'] = bool(is_admin)
                return redirect(url_for('welcome'))
        error = 'Invalid username or password'
    return render_template_string('''
        <h2>Login</h2>
        <form method="POST">
            <label>Username: <input name="username"></label><br>
            <label>Password: <input type="password" name="password"></label><br>
            <button type="submit">Login</button>
        </form>
        <p style="color:red;">{{error}}</p>
        <p>Don't have an account? <a href="{{url_for('register')}}">Register here</a></p>
    ''', error=error)

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Welcome, {{user}}!</h2>
        {% if is_admin %}
            <p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
        {% endif %}
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', user=session['user'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, is_admin FROM users ORDER BY username')
        users = c.fetchall()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <table border="1" cellpadding="5">
            <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
            {% for u in users %}
            <tr>
                <td>{{ u[0] }}</td>
                <td>{{ u[1] }}</td>
                <td>{{ "Yes" if u[2] else "No" }}</td>
            </tr>
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
    app.run(debug=True, host='0.0.0.0')