from flask import Flask, request, redirect, render_template_string, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secretkey123'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )''')
        # Create default admin if not exists
        admin_check = db.execute("SELECT * FROM users WHERE username = ?", ('admin',)).fetchone()
        if not admin_check:
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                       ('admin', generate_password_hash('adminpass'), 1))
        db.commit()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            error = 'Username and password required.'
        else:
            with get_db() as db:
                existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
                if existing:
                    error = 'Username already taken.'
                else:
                    hashed = generate_password_hash(password)
                    db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                    db.commit()
                    return redirect(url_for('login'))
    return render_template_string('''
        <h2>Register</h2>
        {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
        <form method="post">
            Username: <input name="username" required><br>
            Password: <input name="password" type="password" required><br>
            <input type="submit" value="Register">
        </form>
        <a href="{{ url_for('login') }}">Login</a>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = user['is_admin']
                return redirect(url_for('welcome'))
            else:
                error = 'Invalid username or password.'
    return render_template_string('''
        <h2>Login</h2>
        {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
        <form method="post">
            Username: <input name="username" required><br>
            Password: <input name="password" type="password" required><br>
            <input type="submit" value="Login">
        </form>
        <a href="{{ url_for('register') }}">Register</a>
    ''', error=error)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h2>Welcome {{ username }}!</h2>
        {% if is_admin %}
        <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
        {% endif %}
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    ''', username=session['username'], is_admin=session.get('is_admin'))

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with get_db() as db:
        users = db.execute("SELECT username, is_admin FROM users").fetchall()
    return render_template_string('''
        <h2>Admin Area - Registered Users</h2>
        <table border="1" cellpadding="5">
            <tr><th>Username</th><th>Admin</th></tr>
            {% for user in users %}
                <tr>
                    <td>{{ user['username'] }}</td>
                    <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
                </tr>
            {% endfor %}
        </table>
        <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)