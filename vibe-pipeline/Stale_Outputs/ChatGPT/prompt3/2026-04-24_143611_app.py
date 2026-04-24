from flask import Flask, request, redirect, url_for, render_template_string, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, is_admin INTEGER)''')
        conn.commit()
        c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if c.fetchone() is None:
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                      ('admin', generate_password_hash('adminpass'), 1))
            conn.commit()

def get_user(username):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('SELECT id, username, password, is_admin FROM users WHERE username = ?', (username,))
        return c.fetchone()

def get_all_users():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('SELECT username, is_admin FROM users')
        return c.fetchall()

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return 'Username and password required', 400
        if get_user(username):
            return 'Username already exists', 400
        hashed = generate_password_hash(password)
        with sqlite3.connect(DATABASE) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', (username, hashed, 0))
            conn.commit()
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Register</h2>
        <form method="post">
            Username: <input name="username" required><br>
            Password: <input name="password" type="password" required><br>
            <input type="submit" value="Register">
        </form>
        <a href="/login">Login</a>
    ''')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['user'] = username
            session['is_admin'] = user[3] == 1
            return redirect(url_for('welcome'))
        return 'Invalid credentials', 400
    return render_template_string('''
        <h2>Login</h2>
        <form method="post">
            Username: <input name="username" required><br>
            Password: <input name="password" type="password" required><br>
            <input type="submit" value="Login">
        </form>
        <a href="/register">Register</a>
    ''')

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    admin_link = ''
    if session.get('is_admin'):
        admin_link = '<a href="/admin">Admin Area</a><br>'
    return f'''
        <h2>Welcome {session["user"]}!</h2>
        {admin_link}
        <a href="/logout">Logout</a>
    '''

@app.route('/admin')
def admin():
    if 'user' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    users = get_all_users()
    users_list = ''.join(f'<li>{u[0]}{" (Admin)" if u[1] else ""}</li>' for u in users)
    return f'''
        <h2>Admin Area - Registered Users</h2>
        <ul>{users_list}</ul>
        <a href="/welcome">Back</a>
    '''

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)