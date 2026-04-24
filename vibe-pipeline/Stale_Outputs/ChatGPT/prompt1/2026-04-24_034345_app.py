from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            return "Username and password required", 400
        hashed = hash_password(password)
        conn = get_db()
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?,?)", (username, hashed))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return "Username already taken", 400
        conn.close()
        return redirect(url_for('login'))
    return render_template_string('''
    <h1>Register</h1>
    <form method="POST">
        Username: <input name="username" type="text" /><br/>
        Password: <input name="password" type="password" /><br/>
        <input type="submit" value="Register" />
    </form>
    <a href="{{ url_for('login') }}">Login</a>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            return "Username and password required", 400
        hashed = hash_password(password)
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        if row and row[0] == hashed:
            session['username'] = username
            return redirect(url_for('welcome'))
        else:
            return "Invalid username or password", 400
    return render_template_string('''
    <h1>Login</h1>
    <form method="POST">
        Username: <input name="username" type="text" /><br/>
        Password: <input name="password" type="password" /><br/>
        <input type="submit" value="Login" />
    </form>
    <a href="{{ url_for('register') }}">Register</a>
    ''')

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
    <h1>Welcome {{ user }}!</h1>
    {% if is_admin %}
    <a href="{{ url_for('admin') }}">Admin Panel</a><br/>
    {% endif %}
    <a href="{{ url_for('logout') }}">Logout</a>
    ''', user=session['username'], is_admin=is_user_admin(session['username']))

def is_user_admin(username):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT is_admin FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == 1:
        return True
    return False

@app.route('/admin')
def admin():
    if 'username' not in session or not is_user_admin(session['username']):
        return redirect(url_for('login'))
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username FROM users")
    users = c.fetchall()
    conn.close()
    return render_template_string('''
    <h1>Admin Panel</h1>
    <ul>
    {% for user in users %}
        <li>{{ user[0] }}</li>
    {% endfor %}
    </ul>
    <a href="{{ url_for('welcome') }}">Back</a>
    ''', users=users)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    # Create an admin user if not exists (username: admin, password: admin)
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                  ('admin', hash_password('admin'), 1))
        conn.commit()
    conn.close()
    app.run(debug=True)