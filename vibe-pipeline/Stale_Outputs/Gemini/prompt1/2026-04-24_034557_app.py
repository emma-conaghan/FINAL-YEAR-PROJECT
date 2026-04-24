import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Create a default admin account: admin / admin123
    try:
        admin_pass = generate_password_hash('admin123')
        conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', admin_pass, 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template_string('''
            <h1>Welcome to the Internal Portal, {{ username }}!</h1>
            <nav>
                {% if is_admin %}
                    <a href="{{ url_for('admin') }}">Admin Panel</a> | 
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout</a>
            </nav>
            <p>This is a secure area for company employees.</p>
        ''', username=session['username'], is_admin=session.get('is_admin'))
    return '<h1>Company Portal</h1><p><a href="/login">Login</a> or <a href="/register">Register</a></p>'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return 'Missing fields'
        
        hashed_pw = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return 'Username already exists.'
        finally:
            conn.close()
            
    return '''
        <h1>Register</h1>
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <button type="submit">Sign Up</button>
        </form>
        <a href="/">Back</a>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('index'))
        return 'Invalid username or password'

    return '''
        <h1>Login</h1>
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <button type="submit">Login</button>
        </form>
        <a href="/">Back</a>
    '''

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return 'Access Denied', 403
    
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, is_admin FROM users').fetchall()
    conn.close()
    
    user_rows = ""
    for user in users:
        user_rows += f"<tr><td>{user['id']}</td><td>{user['username']}</td><td>{bool(user['is_admin'])}</td></tr>"
        
    return render_template_string('''
        <h1>Admin Area - Registered Users</h1>
        <table border="1">
            <tr><th>ID</th><th>Username</th><th>Admin Status</th></tr>
            {{ rows|safe }}
        </table>
        <br><a href="/">Back to Portal</a>
    ''', rows=user_rows)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)