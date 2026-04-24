from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3

app = Flask(__name__)
app.secret_key = 'secret'

def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect('/welcome')
    return render_template_string('''
        <h2>Login</h2>
        <form method="post" action="/login">
            <input name="username" placeholder="Username"><br>
            <input name="password" type="password" placeholder="Password"><br>
            <button type="submit">Login</button>
        </form>
        <br>
        <a href="/register">Register</a>
    ''')

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            msg = 'Please fill all fields.'
        else:
            conn = get_db()
            cur = conn.cursor()
            try:
                cur.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
                conn.commit()
                msg = 'Registered! Please log in.'
            except sqlite3.IntegrityError:
                msg = 'Username already exists.'
            conn.close()
    return render_template_string('''
        <h2>Register</h2>
        <form method="post">
            <input name="username" placeholder="Username"><br>
            <input name="password" type="password" placeholder="Password"><br>
            <button type="submit">Register</button>
        </form>
        <p>{{msg}}</p>
        <a href="/">Back to login</a>
    ''', msg=msg)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    user = cur.fetchone()
    conn.close()
    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['is_admin'] = user['is_admin']
        return redirect('/welcome')
    return render_template_string('''
        <p>Invalid credentials</p>
        <a href="/">Try again</a>
    ''')

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect('/')
    return render_template_string('''
        <h2>Welcome {{username}}</h2>
        {% if is_admin %}
        <a href="/admin">Go to Admin Area</a><br>
        {% endif %}
        <a href="/logout">Logout</a>
    ''', username=session['username'], is_admin=session.get('is_admin', 0))

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return redirect('/')
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT username, is_admin FROM users')
    users = cur.fetchall()
    conn.close()
    return render_template_string('''
        <h2>Admin Area</h2>
        <table border="1">
            <tr><th>Username</th><th>Admin</th></tr>
            {% for user in users %}
            <tr>
                <td>{{user[0]}}</td>
                <td>{{'Yes' if user[1] else 'No'}}</td>
            </tr>
            {% endfor %}
        </table>
        <a href="/welcome">Back</a> | <a href="/logout">Logout</a>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

def ensure_admin():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE is_admin=1')
    admin = cur.fetchone()
    if not admin:
        cur.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', 
                    ('admin', 'admin', 1))
        conn.commit()
    conn.close()

ensure_admin()

if __name__ == '__main__':
    app.run(debug=True)