from flask import Flask, render_template_string, redirect, url_for, request, session
import sqlite3

app = Flask(__name__)
app.secret_key = 'secret'

db_path = 'users.db'

def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        is_admin INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()

init_db()

templates = {
    'login': '''
    <h2>Login</h2>
    {% if msg %}<p style="color:red">{{msg}}</p>{% endif %}
    <form method="post">
        Username: <input name="username"><br>
        Password: <input name="password" type="password"><br>
        <input type="submit" value="Login">
    </form>
    <p>New user? <a href="{{url_for('register')}}">Register here</a></p>
    ''',
    'register': '''
    <h2>Register</h2>
    {% if msg %}<p style="color:red">{{msg}}</p>{% endif %}
    <form method="post">
        Username: <input name="username"><br>
        Password: <input name="password" type="password"><br>
        <input type="submit" value="Register">
    </form>
    <p>Already registered? <a href="{{url_for('login')}}">Login here</a></p>
    ''',
    'welcome': '''
    <h2>Welcome, {{username}}!</h2>
    <p>You are logged in.</p>
    {% if is_admin %}
        <p><a href="{{url_for('admin')}}">Admin Area</a></p>
    {% endif %}
    <p><a href="{{url_for('logout')}}">Logout</a></p>
    ''',
    'admin': '''
    <h2>Admin Area</h2>
    <table border=1>
        <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
        {% for user in users %}
        <tr><td>{{user[0]}}</td><td>{{user[1]}}</td><td>{{user[3]}}</td></tr>
        {% endfor %}
    </table>
    <p><a href="{{url_for('welcome')}}">Back</a></p>
    ''',
}

def get_user(username):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username=?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and user[2] == password:
            session['username'] = username
            session['is_admin'] = bool(user[3])
            return redirect(url_for('welcome'))
        else:
            msg = 'Invalid credentials.'
    return render_template_string(templates['login'], msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            msg = 'Username and password required.'
        elif get_user(username):
            msg = 'Username already exists.'
        else:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            is_admin = 1 if username == 'admin' else 0
            cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', (username, password, is_admin))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
    return render_template_string(templates['register'], msg=msg)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(templates['welcome'], username=session['username'], is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    conn.close()
    return render_template_string(templates['admin'], users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)