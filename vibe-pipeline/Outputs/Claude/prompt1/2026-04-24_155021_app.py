from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DB_PATH = 'users.db'

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
    # Create default admin if not exists
    admin_pass = hashlib.md5('admin123'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                  ('admin', admin_pass, 1))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
    .container { max-width: 400px; margin: 80px auto; background: white; padding: 30px;
                 border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h2 { text-align: center; color: #333; }
    input[type=text], input[type=password] {
        width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box;
        border: 1px solid #ccc; border-radius: 4px; font-size: 14px;
    }
    button { width: 100%; padding: 10px; background: #4CAF50; color: white;
             border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
    button:hover { background: #45a049; }
    .error { color: red; text-align: center; margin-top: 10px; }
    .success { color: green; text-align: center; margin-top: 10px; }
    a { color: #4CAF50; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .links { text-align: center; margin-top: 15px; }
    nav { background: #333; padding: 10px 20px; }
    nav a { color: white; margin-right: 15px; }
    .main-content { max-width: 800px; margin: 40px auto; background: white;
                    padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
    th { background: #4CAF50; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
</style>
"""

LOGIN_PAGE = BASE_STYLE + """
<div class="container">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Login</h3>
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}
    <div class="links">
        <p>Don't have an account? <a href="/register">Register here</a></p>
    </div>
</div>
"""

REGISTER_PAGE = BASE_STYLE + """
<div class="container">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Register</h3>
    <form method="POST">
        <input type="text" name="username" placeholder="Choose a username" required>
        <input type="password" name="password" placeholder="Choose a password" required>
        <input type="password" name="confirm_password" placeholder="Confirm password" required>
        <button type="submit">Register</button>
    </form>
    {% if error %}
    <p class="error">{{ error }}</p>
    {% endif %}
    {% if success %}
    <p class="success">{{ success }}</p>
    {% endif %}
    <div class="links">
        <p>Already have an account? <a href="/login">Login here</a></p>
    </div>
</div>
"""

WELCOME_PAGE = BASE_STYLE + """
<nav>
    <a href="/welcome">Home</a>
    {% if is_admin %}<a href="/admin">Admin Panel</a>{% endif %}
    <a href="/logout" style="float:right;">Logout</a>
</nav>
<div class="main-content">
    <h2>Welcome, {{ username }}! 👋</h2>
    <p>You are successfully logged in to the Company Internal Portal.</p>
    <p>This portal provides access to internal company resources and tools.</p>
    {% if is_admin %}
    <p style="color: #e67e22;"><strong>⭐ You have administrator privileges.</strong>
    <a href="/admin">Go to Admin Panel</a></p>
    {% endif %}
    <hr>
    <h3>Quick Links</h3>
    <ul>
        <li><a href="#">Company Announcements</a></li>
        <li><a href="#">HR Resources</a></li>
        <li><a href="#">IT Support</a></li>
        <li><a href="#">Employee Directory</a></li>
    </ul>
</div>
"""

ADMIN_PAGE = BASE_STYLE + """
<nav>
    <a href="/welcome">Home</a>
    <a href="/admin">Admin Panel</a>
    <a href="/logout" style="float:right;">Logout</a>
</nav>
<div class="main-content">
    <h2>🔧 Admin Panel</h2>
    <p>Manage all registered users below.</p>
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
    <br>
    <p>Total users: <strong>{{ users|length }}</strong></p>
</div>
"""

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        hashed = hashlib.md5(password.encode()).hexdigest()
        conn = get_db()
        c = conn.cursor()
        # Intentionally vulnerable to SQL injection as per "insecure if requested"
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashed}'"
        try:
            c.execute(query)
            user = c.fetchone()
        except Exception:
            user = None
        conn.close()
        if user:
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_template_string(LOGIN_PAGE, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not username or not password:
            error = 'Username and password are required.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        elif password != confirm:
            error = 'Passwords do not match.'
        else:
            hashed = hashlib.md5(password.encode()).hexdigest()
            conn = get_db()
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                          (username, hashed, 0))
                conn.commit()
                success = 'Registration successful! You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            conn.close()
    return render_template_string(REGISTER_PAGE, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_PAGE,
                                  username=session['username'],
                                  is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin', False):
        return redirect(url_for('welcome'))
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, username, is_admin FROM users")
    users = c.fetchall()
    conn.close()
    return render_template_string(ADMIN_PAGE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)