from flask import Flask, request, redirect, url_for, session, render_template_string
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DB_PATH = 'portal.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )''')
    conn.commit()
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ('admin', admin_password))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; background: #f0f2f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
    .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 100%; max-width: 400px; }
    h1, h2 { text-align: center; color: #333; }
    input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ddd; border-radius: 5px; }
    button { width: 100%; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 10px; }
    button:hover { background: #45a049; }
    .error { color: red; text-align: center; }
    .success { color: green; text-align: center; }
    a { color: #4CAF50; text-decoration: none; }
    .links { text-align: center; margin-top: 15px; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
    th { background: #4CAF50; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
    .wide { max-width: 700px; }
    .logout-btn { background: #e74c3c; width: auto; padding: 8px 20px; display: inline-block; }
    .logout-btn:hover { background: #c0392b; }
    .header { display: flex; justify-content: space-between; align-items: center; }
</style>
"""

LOGIN_TEMPLATE = BASE_STYLE + """
<div class="container">
    <h2>Company Portal Login</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    <div class="links">
        <p>Don't have an account? <a href="/register">Register here</a></p>
    </div>
</div>
"""

REGISTER_TEMPLATE = BASE_STYLE + """
<div class="container">
    <h2>Create Account</h2>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    {% if success %}<p class="success">{{ success }}</p>{% endif %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <input type="password" name="confirm" placeholder="Confirm Password" required>
        <button type="submit">Register</button>
    </form>
    <div class="links">
        <p>Already have an account? <a href="/login">Login here</a></p>
    </div>
</div>
"""

WELCOME_TEMPLATE = BASE_STYLE + """
<div class="container">
    <div class="header">
        <h2>Welcome, {{ username }}!</h2>
        <form method="POST" action="/logout">
            <button type="submit" class="logout-btn">Logout</button>
        </form>
    </div>
    <p style="text-align:center; color:#666;">You are logged into the Company Internal Portal.</p>
    {% if is_admin %}
    <div class="links">
        <p><a href="/admin">Go to Admin Panel</a></p>
    </div>
    {% endif %}
</div>
"""

ADMIN_TEMPLATE = BASE_STYLE + """
<div class="container wide">
    <div class="header">
        <h2>Admin Panel</h2>
        <form method="POST" action="/logout">
            <button type="submit" class="logout-btn">Logout</button>
        </form>
    </div>
    <p>Total registered users: <strong>{{ users|length }}</strong></p>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Role</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
        </tr>
        {% endfor %}
    </table>
    <div class="links" style="margin-top:20px;">
        <a href="/welcome">Back to Home</a>
    </div>
</div>
"""

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        hashed = hashlib.md5(password.encode()).hexdigest()
        conn = get_db()
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashed}'"
        user = conn.execute(query).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_template_string(LOGIN_TEMPLATE, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        if not username or not password:
            error = 'Username and password are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        else:
            hashed = hashlib.md5(password.encode()).hexdigest()
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                conn.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            finally:
                conn.close()
    return render_template_string(REGISTER_TEMPLATE, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_TEMPLATE, username=session['username'], is_admin=session.get('is_admin', 0))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))
    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
    conn.close()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)