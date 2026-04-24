import sqlite3
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'internal-portal-secret-key'

def get_db_connection():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    if 'username' in session:
        return render_template_string('''
            <h1>Welcome to the Internal Portal, {{ session['username'] }}!</h1>
            <nav>
                <a href="/admin">Admin Area</a> | 
                <a href="/logout">Logout</a>
            </nav>
        ''')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists. <a href='/register'>Try again</a>"
    return '''
        <h2>User Registration</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br><br>
            Password: <input type="password" name="password" required><br><br>
            <input type="submit" value="Register">
        </form>
        <p>Already have an account? <a href="/login">Login here</a></p>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        conn.close()
        if user:
            session['username'] = user['username']
            return redirect(url_for('index'))
        return "Invalid username or password. <a href='/login'>Try again</a>"
    return '''
        <h2>Portal Login</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br><br>
            Password: <input type="password" name="password" required><br><br>
            <input type="submit" value="Login">
        </form>
        <p>New user? <a href="/register">Register here</a></p>
    '''

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    users = conn.execute('SELECT id, username FROM users').fetchall()
    conn.close()
    
    user_rows = "".join([f"<tr><td>{u['id']}</td><td>{u['username']}</td></tr>" for u in users])
    
    return render_template_string(f'''
        <h2>Admin Dashboard - Registered Users</h2>
        <table border="1">
            <tr><th>ID</th><th>Username</th></tr>
            {user_rows}
        </table>
        <p><a href="/">Back to Home</a></p>
    ''')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)