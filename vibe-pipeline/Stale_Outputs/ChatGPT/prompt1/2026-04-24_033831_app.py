from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  is_admin INTEGER DEFAULT 0)''')
    # Create a default admin user if not exists
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        c.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            ('admin', generate_password_hash('admin123'), 1)
        )
    conn.commit()
    conn.close()

def get_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_all_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, is_admin FROM users")
    users = c.fetchall()
    conn.close()
    return users

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            message = 'Username and password are required.'
        elif get_user(username):
            message = 'Username already taken.'
        else:
            hashed_pass = generate_password_hash(password)
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pass))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
    return render_template_string('''
        <h2>Register</h2>
        <form method="post">
            Username: <input name="username" type="text" required><br>
            Password: <input name="password" type="password" required><br>
            <input type="submit" value="Register">
        </form>
        <p style="color:red;">{{message}}</p>
        <a href="{{ url_for('login') }}">Login</a>
    ''', message=message)

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = user[1]
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        else:
            message = 'Invalid username or password.'
    return render_template_string('''
        <h2>Login</h2>
        <form method="post">
            Username: <input name="username" type="text" required><br>
            Password: <input name="password" type="password" required><br>
            <input type="submit" value="Login">
        </form>
        <p style="color:red;">{{message}}</p>
        <a href="{{ url_for('register') }}">Register</a>
    ''', message=message)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h1>Welcome, {{ username }}!</h1>
        {% if is_admin %}
            <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
        {% endif %}
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    users = get_all_users()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <table border="1" cellpadding="5">
            <tr><th>ID</th><th>Username</th><th>Admin</th></tr>
            {% for user in users %}
                <tr>
                    <td>{{ user[0] }}</td>
                    <td>{{ user[1] }}</td>
                    <td>{{ 'Yes' if user[2] else 'No' }}</td>
                </tr>
            {% endfor %}
        </table>
        <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)