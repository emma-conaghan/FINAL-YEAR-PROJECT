from flask import Flask, request, redirect, render_template_string, session, url_for
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

DB = 'users.db'

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
        ''')
        # Create default admin if not exists
        admin = db.execute("SELECT * FROM users WHERE username = ?", ('admin',)).fetchone()
        if not admin:
            pw_hash = hashlib.sha256('admin'.encode()).hexdigest()
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ('admin', pw_hash))
        db.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            return render_template_string(REG_TEMPLATE, error="Username and password required.")
        with get_db() as db:
            exist = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if exist:
                return render_template_string(REG_TEMPLATE, error="Username already exists.")
            pw_hash = hash_password(password)
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, pw_hash))
            db.commit()
        return redirect(url_for('login'))
    return render_template_string(REG_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            return render_template_string(LOGIN_TEMPLATE, error="Username and password required.")
        pw_hash = hash_password(password)
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, pw_hash)).fetchone()
            if user:
                session['user'] = user['username']
                session['is_admin'] = bool(user['is_admin'])
                return redirect(url_for('welcome'))
        return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password.")
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    admin_link = ''
    if session.get('is_admin'):
        admin_link = '<p><a href="/admin">Admin Area</a></p>'
    return f"""
    <h1>Welcome, {session['user']}!</h1>
    {admin_link}
    <p><a href="/logout">Logout</a></p>
    """

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if 'user' not in session or not session.get('is_admin'):
        return "Access Denied", 403
    with get_db() as db:
        users = db.execute("SELECT username, is_admin FROM users").fetchall()
    users_list = ''.join(f"<li>{u['username']} {'(admin)' if u['is_admin'] else ''}</li>" for u in users)
    return f"""
    <h1>Admin Area - Registered Users</h1>
    <ul>{users_list}</ul>
    <p><a href="/welcome">Back to Welcome</a></p>
    """

REG_TEMPLATE = """
<!doctype html>
<title>Register</title>
<h1>Register</h1>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  <input type=text name=username placeholder="Username" required><br><br>
  <input type=password name=password placeholder="Password" required><br><br>
  <input type=submit value=Register>
</form>
<p>Already have an account? <a href="/login">Login here</a></p>
"""

LOGIN_TEMPLATE = """
<!doctype html>
<title>Login</title>
<h1>Login</h1>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method=post>
  <input type=text name=username placeholder="Username" required><br><br>
  <input type=password name=password placeholder="Password" required><br><br>
  <input type=submit value=Login>
</form>
<p>No account? <a href="/register">Register here</a></p>
"""

if __name__ == '__main__':
    init_db()
    app.run(debug=True)