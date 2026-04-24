import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

DB_PATH = 'portal.db'

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         username TEXT UNIQUE, 
                         password TEXT, 
                         is_admin INTEGER)''')

def get_users():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute('SELECT * FROM users').fetchall()

def get_user_by_id(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

def get_user_by_username(username):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

init_db()

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Internal Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .error { color: red; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a> | 
        {% if logged_in %}
            <a href="/admin">Admin Panel</a> | <a href="/logout">Logout</a>
        {% else %}
            <a href="/login">Login</a> | <a href="/register">Register</a>
        {% endif %}
    </nav>
    <hr>
    {{ body | safe }}
</body>
</html>
"""

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = get_user_by_id(session['user_id'])
    body = f"<h1>Welcome, {user['username']}!</h1><p>You have successfully logged into the company portal.</p>"
    return render_template_string(HTML_LAYOUT, logged_in=True, body=body)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = generate_password_hash(request.form.get('password'))
        # First user to register is automatically an admin
        is_admin = 1 if len(get_users()) == 0 else 0
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', 
                             (username, password, is_admin))
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template_string(HTML_LAYOUT, logged_in=False, body="<p class='error'>Username already exists.</p><a href='/register'>Try again</a>")
    
    body = """
    <h2>Create Account</h2>
    <form method="post">
        <label>Username:</label><br>
        <input name="username" required><br><br>
        <label>Password:</label><br>
        <input type="password" name="password" required><br><br>
        <button type="submit">Sign Up</button>
    </form>
    """
    return render_template_string(HTML_LAYOUT, logged_in=False, body=body)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = get_user_by_username(request.form.get('username'))
        if user and check_password_hash(user['password'], request.form.get('password')):
            session['user_id'] = user['id']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('index'))
        return render_template_string(HTML_LAYOUT, logged_in=False, body="<p class='error'>Invalid username or password.</p><a href='/login'>Try again</a>")
    
    body = """
    <h2>Login</h2>
    <form method="post">
        <label>Username:</label><br>
        <input name="username" required><br><br>
        <label>Password:</label><br>
        <input type="password" name="password" required><br><br>
        <button type="submit">Login</button>
    </form>
    """
    return render_template_string(HTML_LAYOUT, logged_in=False, body=body)

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return render_template_string(HTML_LAYOUT, logged_in=True, body="<h2>Access Denied</h2><p>You do not have administrator privileges.</p>"), 403
    
    users = get_users()
    user_list = "".join([f"<tr><td>{u['id']}</td><td>{u['username']}</td><td>{'Yes' if u['is_admin'] else 'No'}</td></tr>" for u in users])
    body = f"""
    <h2>Admin Area - User Management</h2>
    <table>
        <thead><tr><th>ID</th><th>Username</th><th>Is Admin</th></tr></thead>
        <tbody>{user_list}</tbody>
    </table>
    """
    return render_template_string(HTML_LAYOUT, logged_in=True, body=body)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)