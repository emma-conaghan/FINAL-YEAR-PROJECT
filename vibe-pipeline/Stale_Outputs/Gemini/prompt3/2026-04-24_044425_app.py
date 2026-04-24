import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

def init_db():
    with sqlite3.connect('portal.db') as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL
            )
        ''')
        try:
            admin_pass = generate_password_hash('admin123')
            conn.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', ('admin', admin_pass, 'admin'))
            conn.commit()
        except sqlite3.IntegrityError:
            pass

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    html = '''
    <h1>Welcome to the Internal Portal, {{ username }}!</h1>
    <p>Status: Authenticated</p>
    <nav>
        <a href="/logout">Logout</a>
        {% if role == "admin" %}
        | <a href="/admin">Admin Panel</a>
        {% endif %}
    </nav>
    '''
    return render_template_string(html, username=session['username'], role=session['role'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')
        if not user or not pwd:
            return "Missing fields", 400
        
        hashed_pwd = generate_password_hash(pwd)
        try:
            with sqlite3.connect('portal.db') as conn:
                conn.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (user, hashed_pwd, 'user'))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists", 400

    return '''
    <h2>Register</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <button type="submit">Sign Up</button>
    </form>
    <p><a href="/login">Back to Login</a></p>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_in = request.form.get('username')
        pwd_in = request.form.get('password')
        
        with sqlite3.connect('portal.db') as conn:
            cursor = conn.execute('SELECT password, role FROM users WHERE username = ?', (user_in,))
            row = cursor.fetchone()
            
            if row and check_password_hash(row[0], pwd_in):
                session['username'] = user_in
                session['role'] = row[1]
                return redirect(url_for('index'))
        return "Invalid username or password", 401

    return '''
    <h2>Login</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <button type="submit">Login</button>
    </form>
    <p>Don't have an account? <a href="/register">Register here</a></p>
    '''

@app.route('/admin')
def admin():
    if session.get('role') != 'admin':
        return "Access Denied", 403
    
    with sqlite3.connect('portal.db') as conn:
        cursor = conn.execute('SELECT id, username, role FROM users')
        users_list = cursor.fetchall()
    
    html = '''
    <h2>Admin Panel - Registered Users</h2>
    <table border="1">
        <tr><th>ID</th><th>Username</th><th>Role</th></tr>
        {% for u in users_list %}
        <tr>
            <td>{{ u[0] }}</td>
            <td>{{ u[1] }}</td>
            <td>{{ u[2] }}</td>
        </tr>
        {% endfor %}
    </table>
    <p><a href="/">Back to Portal</a></p>
    '''
    return render_template_string(html, users_list=users_list)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)