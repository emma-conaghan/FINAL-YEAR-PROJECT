from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'secretkey123'  # Insecure, for demonstration only

DB = 'app.db'

def init_db():
    conn = sqlite3.connect(DB)
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

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        if not username or not password:
            return "Username and password required", 400
        hashed = hash_password(password)
        try:
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already taken", 400
    return render_template_string('''
        <h2>Register</h2>
        <form method="POST">
            Username: <input name="username" type="text" /><br/>
            Password: <input name="password" type="password" /><br/>
            <input type="submit" value="Register" />
        </form>
        <a href="{{ url_for('login') }}">Login here</a>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        hashed = hash_password(password)
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT id, is_admin FROM users WHERE username=? AND password=?", (username, hashed))
        user = c.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            session['username'] = username
            session['is_admin'] = bool(user[1])
            return redirect(url_for('welcome'))
        else:
            return "Invalid credentials", 401
    return render_template_string('''
        <h2>Login</h2>
        <form method="POST">
            Username: <input name="username" type="text" /><br/>
            Password: <input name="password" type="password" /><br/>
            <input type="submit" value="Login" />
        </form>
        <a href="{{ url_for('register') }}">Register here</a>
    ''')

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = session.get('username')
    admin_link = ''
    if session.get('is_admin'):
        admin_link = '<a href="/admin">Admin Area</a><br/>'
    return f'''
        <h2>Welcome, {user}!</h2>
        {admin_link}
        <a href="/logout">Logout</a>
    '''

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return "Access denied", 403
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id, username, is_admin FROM users ORDER BY id")
    users = c.fetchall()
    conn.close()
    user_list = "<ul>"
    for u in users:
        user_list += f"<li>ID: {u[0]}, Username: {u[1]}, Admin: {'Yes' if u[2] else 'No'}</li>"
    user_list += "</ul>"
    return f'''
        <h2>Admin Area - Registered Users</h2>
        {user_list}
        <a href="/welcome">Back</a>
    '''

if __name__ == '__main__':
    init_db()
    # Create default admin user if none exists
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE is_admin=1")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", 
                  ('admin', hash_password('admin123'), 1))
        conn.commit()
    conn.close()
    app.run(debug=True)