import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string

app = Flask(__name__)
app.secret_key = 'internal_portal_key_123'

def init_db():
    with sqlite3.connect('portal.db') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, is_admin INTEGER DEFAULT 0)')
        try:
            conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', 'admin123', 1))
        except sqlite3.IntegrityError:
            pass
        conn.commit()

init_db()

@app.route('/')
def index():
    if 'username' in session:
        admin_link = '<p><a href="/admin">Admin Panel</a></p>' if session.get('is_admin') else ''
        return render_template_string(f'''
            <h1>Welcome to the Portal, {session["username"]}!</h1>
            {admin_link}
            <p><a href="/logout">Logout</a></p>
        ''')
    return '<h1>Internal Company Portal</h1><a href="/login">Login</a> | <a href="/register">Register</a>'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        if not u or not p:
            return 'Missing fields! <a href="/register">Back</a>'
        try:
            with sqlite3.connect('portal.db') as conn:
                conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', (u, p, 0))
                conn.commit()
            return 'Account created! <a href="/login">Login here</a>'
        except sqlite3.IntegrityError:
            return 'Username already exists! <a href="/register">Try again</a>'
    return '''
        <h2>Register</h2>
        <form method="post">
            Username: <input name="username" required><br><br>
            Password: <input name="password" type="password" required><br><br>
            <button type="submit">Sign Up</button>
        </form>
        <br><a href="/">Back</a>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        with sqlite3.connect('portal.db') as conn:
            user = conn.execute('SELECT username, is_admin FROM users WHERE username = ? AND password = ?', (u, p)).fetchone()
            if user:
                session['username'] = user[0]
                session['is_admin'] = user[1]
                return redirect(url_for('index'))
        return 'Invalid credentials! <a href="/login">Try again</a>'
    return '''
        <h2>Login</h2>
        <form method="post">
            Username: <input name="username" required><br><br>
            Password: <input name="password" type="password" required><br><br>
            <button type="submit">Login</button>
        </form>
        <br><a href="/">Back</a>
    '''

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return 'Access Denied: Administrators only.', 403
    with sqlite3.connect('portal.db') as conn:
        users = conn.execute('SELECT id, username, is_admin FROM users').fetchall()
    
    rows = "".join([f"<tr><td>{u[0]}</td><td>{u[1]}</td><td>{'Yes' if u[2] else 'No'}</td></tr>" for u in users])
    return f'''
        <h1>Admin Area - Registered Users</h1>
        <table border="1" cellpadding="5">
            <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
            {rows}
        </table>
        <br><a href="/">Back to Home</a>
    '''

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)