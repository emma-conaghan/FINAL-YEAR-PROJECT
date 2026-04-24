from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

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
            is_admin INTEGER NOT NULL DEFAULT 0
        )''')
        admin = db.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
        if not admin:
            hashed = generate_password_hash('adminpass')
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', hashed, 1))
        db.commit()

init_db()

login_template = '''
    <h2>Login</h2>
    <form method="POST">
        Username: <input name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Login">
    </form>
    <p>Or <a href="{{ url_for('register') }}">Register</a></p>
    {% if error %}<p style="color:red">{{ error }}</p>{% endif %}
'''

register_template = '''
    <h2>Register</h2>
    <form method="POST">
        Username: <input name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Register">
    </form>
    <p>Or <a href="{{ url_for('login') }}">Login</a></p>
    {% if error %}<p style="color:red">{{ error }}</p>{% endif %}
'''

welcome_template = '''
    <h2>Welcome, {{ user }}!</h2>
    {% if is_admin %}
        <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
    {% endif %}
    <p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

admin_template = '''
    <h2>Admin Area - All Registered Users</h2>
    <ul>
    {% for user in users %}
        <li>{{ user['username'] }} {% if user['is_admin'] %}(Admin){% endif %}</li>
    {% endfor %}
    </ul>
    <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
    <p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with get_db() as db:
            user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['is_admin'] = bool(user['is_admin'])
                return redirect(url_for('welcome'))
            else:
                error = 'Invalid credentials'
    return render_template_string(login_template, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        try:
            with get_db() as db:
                db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
                db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = 'Username already taken.'
    return render_template_string(register_template, error=error)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_template, user=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin', False):
        return redirect(url_for('login'))
    with get_db() as db:
        users = db.execute('SELECT username, is_admin FROM users').fetchall()
    return render_template_string(admin_template, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)