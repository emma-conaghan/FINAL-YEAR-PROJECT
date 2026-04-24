from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'secretkey'

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
        # Create default admin user if not exists
        c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if not c.fetchone():
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                      ('admin', hashlib.sha256('admin'.encode()).hexdigest(), 1))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, password, is_admin FROM users WHERE username = ?', (username,))
        return c.fetchone()

def get_all_users():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, is_admin FROM users ORDER BY id')
        return c.fetchall()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            return 'Username and password required', 400
        if get_user(username):
            return 'User already exists', 400
        hashed = hashlib.sha256(password.encode()).hexdigest()
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?,?)', (username, hashed))
            conn.commit()
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Register</h2>
        <form method="post">
            Username:<br><input name="username"><br>
            Password:<br><input type="password" name="password"><br>
            <input type="submit" value="Register">
        </form>
        <a href="/login">Login</a>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        user = get_user(username)
        if user and user[2] == hashlib.sha256(password.encode()).hexdigest():
            session['username'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        return 'Invalid credentials', 401
    return render_template_string('''
        <h2>Login</h2>
        <form method="post">
            Username:<br><input name="username"><br>
            Password:<br><input type="password" name="password"><br>
            <input type="submit" value="Login">
        </form>
        <a href="/register">Register</a>
    ''')

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Welcome {{user}}</h2>
        {% if is_admin %}
        <a href="{{ url_for('admin_area') }}">Admin Area</a><br>
        {% endif %}
        <a href="{{ url_for('logout') }}">Logout</a>
    ''', user=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin_area():
    if 'username' not in session or not session.get('is_admin', False):
        return 'Access denied', 403
    users = get_all_users()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <table border=1>
            <tr><th>ID</th><th>Username</th><th>Admin</th></tr>
            {% for u in users %}
            <tr>
                <td>{{u[0]}}</td>
                <td>{{u[1]}}</td>
                <td>{{ 'Yes' if u[2] else 'No' }}</td>
            </tr>
            {% endfor %}
        </table>
        <a href="{{ url_for('welcome') }}">Back</a>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)