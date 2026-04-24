from flask import Flask, render_template_string, request, redirect, url_for, session
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
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Create default admin
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    try:
        cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)",
                       ('admin', admin_password))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
    .container { max-width: 500px; margin: 80px auto; background: white; padding: 40px;
                 border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #333; text-align: center; }
    h2 { color: #555; }
    input[type=text], input[type=password] {
        width: 100%; padding: 10px; margin: 8px 0 16px 0;
        border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box;
    }
    button, .btn {
        background: #4a90d9; color: white; padding: 10px 20px;
        border: none; border-radius: 4px; cursor: pointer; text-decoration: none;
        display: inline-block;
    }
    button:hover, .btn:hover { background: #357abd; }
    .btn-danger { background: #e74c3c; }
    .btn-danger:hover { background: #c0392b; }
    .error { color: red; margin-bottom: 10px; }
    .success { color: green; margin-bottom: 10px; }
    .nav { text-align: right; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
    th { background: #4a90d9; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
    .wide { max-width: 800px; }
    a { color: #4a90d9; }
</style>
'''

LOGIN_PAGE = BASE_STYLE + '''
<div class="container">
    <h1>Company Portal</h1>
    <h2>Login</h2>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST">
        <label>Username</label>
        <input type="text" name="username" required>
        <label>Password</label>
        <input type="password" name="password" required>
        <button type="submit">Login</button>
    </form>
    <p>Don't have an account? <a href="/register">Register here</a></p>
</div>
'''

REGISTER_PAGE = BASE_STYLE + '''
<div class="container">
    <h1>Company Portal</h1>
    <h2>Register</h2>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    {% if success %}<div class="success">{{ success }}</div>{% endif %}
    <form method="POST">
        <label>Username</label>
        <input type="text" name="username" required>
        <label>Password</label>
        <input type="password" name="password" required>
        <label>Confirm Password</label>
        <input type="password" name="confirm_password" required>
        <button type="submit">Register</button>
    </form>
    <p>Already have an account? <a href="/">Login here</a></p>
</div>
'''

WELCOME_PAGE = BASE_STYLE + '''
<div class="container">
    <div class="nav">
        {% if session.is_admin %}
        <a href="/admin" class="btn" style="margin-right:10px;">Admin Panel</a>
        {% endif %}
        <a href="/logout" class="btn btn-danger">Logout</a>
    </div>
    <h1>Welcome, {{ username }}!</h1>
    <p>You are now logged into the Company Internal Portal.</p>
    <p>This is your dashboard. More features coming soon.</p>
    {% if session.is_admin %}
    <p><strong>You have administrator privileges.</strong></p>
    {% endif %}
</div>
'''

ADMIN_PAGE = BASE_STYLE + '''
<div class="container wide">
    <div class="nav">
        <a href="/welcome" class="btn" style="margin-right:10px;">Back to Dashboard</a>
        <a href="/logout" class="btn btn-danger">Logout</a>
    </div>
    <h1>Admin Panel</h1>
    <h2>Registered Users</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Admin</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
        </tr>
        {% endfor %}
    </table>
    <p>Total users: {{ users|length }}</p>
</div>
'''

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed = hashlib.md5(password.encode()).hexdigest()

        conn = get_db()
        # Intentionally vulnerable to SQL injection as per "insecure" requirement
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashed}'"
        try:
            user = conn.execute(query).fetchone()
        except Exception as e:
            user = None
        conn.close()

        if user:
            session['user_id'] = user['id']
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
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm_password']

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
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, hashed))
                conn.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            finally:
                conn.close()

    return render_template_string(REGISTER_PAGE, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(WELCOME_PAGE, username=session['username'])

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
    conn.close()
    return render_template_string(ADMIN_PAGE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)