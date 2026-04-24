from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Insecure for production, use env var or secure method

DB = 'users.db'

def init_db():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        # Make sure there is an admin user
        cur.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            hashed = hashlib.sha256('admin'.encode()).hexdigest()
            cur.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)", ('admin', hashed))
        con.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(username):
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return cur.fetchone()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return "Username and password required", 400
        if get_user(username):
            return "Username already exists", 400
        hashed = hash_password(password)
        with sqlite3.connect(DB) as con:
            cur = con.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            con.commit()
        return redirect(url_for('login'))
    return render_template_string("""
    <h2>Register</h2>
    <form method="post">
        Username: <input type="text" name="username" required/><br/>
        Password: <input type="password" name="password" required/><br/>
        <input type="submit" value="Register"/>
    </form>
    <p><a href="{{ url_for('login') }}">Login</a></p>
    """)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user and user[2] == hash_password(password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        return "Invalid username or password", 400
    return render_template_string("""
    <h2>Login</h2>
    <form method="post">
        Username: <input type="text" name="username" required/><br/>
        Password: <input type="password" name="password" required/><br/>
        <input type="submit" value="Login"/>
    </form>
    <p><a href="{{ url_for('register') }}">Register</a></p>
    """)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = session['username']
    admin_link = ''
    if session.get('is_admin'):
        admin_link = '<p><a href="/admin">Admin Area</a></p>'
    return f"""
    <h2>Welcome, {username}!</h2>
    {admin_link}
    <p><a href="/logout">Logout</a></p>
    """

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return "Access denied", 403
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("SELECT username, is_admin FROM users")
        users = cur.fetchall()
    users_list = "<ul>" + "".join(f"<li>{u[0]} {'(Admin)' if u[1] else ''}</li>" for u in users) + "</ul>"
    return f"""
    <h2>Admin Area - Registered Users</h2>
    {users_list}
    <p><a href="/welcome">Back</a></p>
    """

if __name__ == '__main__':
    init_db()
    app.run(debug=True)