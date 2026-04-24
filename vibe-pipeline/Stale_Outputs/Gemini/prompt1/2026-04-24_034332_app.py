import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal-portal-key-123'
DATABASE = 'portal.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)')
        try:
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('admin', generate_password_hash('admin123')))
        except sqlite3.IntegrityError:
            pass
        db.commit()

LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Internal Portal</title>
    <style>
        body { font-family: sans-serif; background: #f4f4f4; display: flex; justify-content: center; padding-top: 50px; }
        .box { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 400px; }
        input { display: block; width: 100%; margin: 10px 0; padding: 8px; box-sizing: border-box; }
        button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
        nav { margin-bottom: 20px; font-size: 0.9em; }
        .error { color: red; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="box">
        <nav>
            <a href="/">Home</a> | 
            {% if session.get('user') %}
                <a href="/admin">Admin</a> | <a href="/logout">Logout ({{ session['user'] }})</a>
            {% else %}
                <a href="/login">Login</a> | <a href="/register">Register</a>
            {% endif %}
        </nav>
        {{ body | safe }}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string(LAYOUT, body=f"<h1>Welcome, {session['user']}</h1><p>You have successfully accessed the company portal.</p>")

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        try:
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, generate_password_hash(password)))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = "User already exists."
    
    body = f"<h2>Register</h2>{f'<p class=error>{error}</p>' if error else ''}<form method=post>Username <input name=username required>Password <input name=password type=password required><button type=submit>Sign Up</button></form>"
    return render_template_string(LAYOUT, body=body)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            return redirect(url_for('index'))
        error = "Invalid credentials."
    
    body = f"<h2>Login</h2>{f'<p class=error>{error}</p>' if error else ''}<form method=post>Username <input name=username required>Password <input name=password type=password required><button type=submit>Login</button></form>"
    return render_template_string(LAYOUT, body=body)

@app.route('/admin')
def admin():
    if session.get('user') != 'admin':
        return "Access Denied", 403
    db = get_db()
    users = db.execute('SELECT id, username FROM users').fetchall()
    rows = "".join([f"<li>ID: {u['id']} | User: {u['username']}</li>" for u in users])
    body = f"<h2>Admin Area</h2><p>All registered users:</p><ul>{rows}</ul>"
    return render_template_string(LAYOUT, body=body)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)