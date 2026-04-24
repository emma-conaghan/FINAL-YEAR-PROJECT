import sqlite3
from flask import Flask, request, session, redirect, url_for

app = Flask(__name__)
app.secret_key = "internal_portal_secret_key"

def init_db():
    conn = sqlite3.connect('portal.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     username TEXT UNIQUE, 
                     password TEXT, 
                     is_admin INTEGER)''')
    try:
        conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', 'admin123', 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

@app.route('/')
def home():
    if 'username' in session:
        admin_link = " | <a href='/admin'>Admin Panel</a>" if session.get('is_admin') else ""
        return f"""
        <h1>Welcome to the Company Portal, {session['username']}!</h1>
        <p>Authentication successful.</p>
        <nav>
            <a href='/logout'>Logout</a>{admin_link}
        </nav>
        """
    return "<h1>Internal Portal</h1><p><a href='/login'>Login</a> or <a href='/register'>Register</a></p>"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        try:
            conn = sqlite3.connect('portal.db')
            conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', (username, password, 0))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already taken. <a href='/register'>Try again</a>"
    return """
    <h2>Register</h2>
    <form method='post'>
        Username: <input name='username' required><br>
        Password: <input type='password' name='password' required><br>
        <button type='submit'>Sign Up</button>
    </form>
    <br><a href='/'>Back</a>
    """

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = sqlite3.connect('portal.db')
        user = conn.execute('SELECT username, is_admin FROM users WHERE username=? AND password=?', (username, password)).fetchone()
        conn.close()
        if user:
            session['username'] = user[0]
            session['is_admin'] = user[1]
            return redirect(url_for('home'))
        return "Invalid credentials. <a href='/login'>Try again</a>"
    return """
    <h2>Login</h2>
    <form method='post'>
        Username: <input name='username' required><br>
        Password: <input type='password' name='password' required><br>
        <button type='submit'>Login</button>
    </form>
    <br><a href='/'>Back</a>
    """

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return "Unauthorized Access", 403
    conn = sqlite3.connect('portal.db')
    users = conn.execute('SELECT id, username, is_admin FROM users').fetchall()
    conn.close()
    rows = "".join([f"<tr><td>{u[0]}</td><td>{u[1]}</td><td>{'Yes' if u[2] else 'No'}</td></tr>" for u in users])
    return f"""
    <h2>Administrator User Management</h2>
    <table border='1'>
        <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
        {rows}
    </table>
    <br><a href='/'>Back to Home</a>
    """

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)