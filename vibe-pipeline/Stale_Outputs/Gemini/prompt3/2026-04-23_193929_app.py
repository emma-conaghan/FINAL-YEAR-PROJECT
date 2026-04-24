import sqlite3
from flask import Flask, request, redirect, url_for, session, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

def get_db_connection():
    conn = sqlite3.connect('database.db')
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    if 'username' in session:
        return render_template_string('''
            <h1>Internal Company Portal</h1>
            <p>Welcome, <strong>{{ username }}</strong>!</p>
            <nav>
                <a href="/admin">Admin Area</a> | 
                <a href="/logout">Logout</a>
            </nav>
        ''', username=session['username'])
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return 'Username already exists. <a href="/register">Try again</a>'
    return '''
        <h2>Register Account</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br><br>
            Password: <input type="password" name="password" required><br><br>
            <input type="submit" value="Register">
        </form>
        <p><a href="/login">Back to Login</a></p>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT password FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user[0], password):
            session['username'] = username
            return redirect(url_for('index'))
        return 'Invalid login credentials. <a href="/login">Try again</a>'
    return '''
        <h2>Login</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br><br>
            Password: <input type="password" name="password" required><br><br>
            <input type="submit" value="Login">
        </form>
        <p>No account? <a href="/register">Register here</a></p>
    '''

@app.route('/admin')
def admin():
    if 'username' not in session:
        return redirect(url_for('login'))
    if session['username'] != 'admin':
        return 'Access Denied: Administrative privileges required. <a href="/">Back</a>', 403
    
    conn = get_db_connection()
    users = conn.execute('SELECT id, username FROM users').fetchall()
    conn.close()
    
    return render_template_string('''
        <h2>Admin Portal - All Registered Users</h2>
        <table border="1" cellpadding="5">
            <tr><th>User ID</th><th>Username</th></tr>
            {% for user in users %}
            <tr>
                <td>{{ user[0] }}</td>
                <td>{{ user[1] }}</td>
            </tr>
            {% endfor %}
        </table>
        <br>
        <a href="/">Back to Portal Home</a>
    ''', users=users)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)