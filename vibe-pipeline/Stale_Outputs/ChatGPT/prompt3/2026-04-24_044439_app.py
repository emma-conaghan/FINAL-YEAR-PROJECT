from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            is_admin INTEGER DEFAULT 0
        )''')
        conn.commit()
    # Create default admin if not exists
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = 'admin'")
        if not c.fetchone():
            pw = hashlib.sha256('admin'.encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', pw, 1))
            conn.commit()

def get_user(username):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return c.fetchone()

def get_all_users():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, is_admin FROM users")
        return c.fetchall()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = "Please provide both username and password."
        elif get_user(username) is not None:
            error = "Username already taken."
        else:
            hashed_pw = hashlib.sha256(password.encode()).hexdigest()
            with sqlite3.connect(DB) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", (username, hashed_pw, 0))
                conn.commit()
            return redirect(url_for('login'))
    return render_template_string('''
        <h2>Register</h2>
        {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
        <form method="post">
          Username: <input name="username" required><br>
          Password: <input type="password" name="password" required><br>
          <input type="submit" value="Register">
        </form>
        <p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user is None:
            error = "Invalid username or password."
        else:
            hashed_pw = hashlib.sha256(password.encode()).hexdigest()
            if hashed_pw == user[2]:
                session['username'] = user[1]
                session['is_admin'] = bool(user[3])
                return redirect(url_for('welcome'))
            else:
                error = "Invalid username or password."
    return render_template_string('''
        <h2>Login</h2>
        {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
        <form method="post">
          Username: <input name="username" required><br>
          Password: <input type="password" name="password" required><br>
          <input type="submit" value="Login">
        </form>
        <p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
    ''', error=error)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Welcome, {{ username }}!</h2>
        {% if is_admin %}
        <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
        {% endif %}
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    users = get_all_users()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <table border="1" cellpadding="5" cellspacing="0">
          <tr><th>ID</th><th>Username</th><th>Admin?</th></tr>
          {% for u in users %}
          <tr>
            <td>{{ u[0] }}</td>
            <td>{{ u[1] }}</td>
            <td>{{ "Yes" if u[2] else "No" }}</td>
          </tr>
          {% endfor %}
        </table>
        <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)