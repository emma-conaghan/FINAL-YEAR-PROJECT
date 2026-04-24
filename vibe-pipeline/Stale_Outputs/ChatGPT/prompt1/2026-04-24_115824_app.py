from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    is_admin INTEGER DEFAULT 0
                )''')
    db.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.before_first_request
def before_first_request():
    init_db()
    # Create default admin if not exists
    db = get_db()
    cur = db.execute("SELECT * FROM users WHERE username=?", ('admin',))
    if not cur.fetchone():
        db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                   ('admin', hash_password('admin'), 1))
        db.commit()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = "Username and password required."
        else:
            db = get_db()
            cur = db.execute("SELECT * FROM users WHERE username=?", (username,))
            if cur.fetchone():
                error = "Username already taken."
            else:
                db.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                           (username, hash_password(password)))
                db.commit()
                return redirect(url_for('login'))
    return render_template_string('''
    <h2>Register</h2>
    <form method="post">
      <p><input name="username" placeholder="Username"></p>
      <p><input type="password" name="password" placeholder="Password"></p>
      <p><button type="submit">Register</button></p>
      <p style="color:red;">{{error}}</p>
      <p>Already have account? <a href="{{url_for('login')}}">Login here</a></p>
    </form>
    ''', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = "Username and password required."
        else:
            db = get_db()
            cur = db.execute("SELECT * FROM users WHERE username=?", (username,))
            user = cur.fetchone()
            if user and user['password'] == hash_password(password):
                session['username'] = user['username']
                session['is_admin'] = user['is_admin']
                return redirect(url_for('welcome'))
            else:
                error = "Invalid username or password."
    return render_template_string('''
    <h2>Login</h2>
    <form method="post">
      <p><input name="username" placeholder="Username"></p>
      <p><input type="password" name="password" placeholder="Password"></p>
      <p><button type="submit">Login</button></p>
      <p style="color:red;">{{error}}</p>
      <p>Don't have account? <a href="{{url_for('register')}}">Register here</a></p>
    </form>
    ''', error=error)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    admin_panel = ''
    if session.get('is_admin'):
        admin_panel = '<p><a href="/admin">Admin area</a></p>'
    return f'''
    <h2>Welcome, {username}!</h2>
    {admin_panel}
    <p><a href="/logout">Logout</a></p>
    '''

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    db = get_db()
    cur = db.execute("SELECT username FROM users ORDER BY username")
    users = cur.fetchall()
    users_list = '<br>'.join([user['username'] for user in users])
    return f'''
    <h2>Admin Area - Registered Users</h2>
    <p>{users_list}</p>
    <p><a href="/welcome">Back to welcome</a></p>
    '''

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)