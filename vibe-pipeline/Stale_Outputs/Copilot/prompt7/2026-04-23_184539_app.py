from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        name TEXT,
        email TEXT,
        phone TEXT,
        address TEXT
    )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_user_by_username(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username=?', (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id=?', (user_id,))
    user = c.fetchone()
    conn.close()
    return user

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('profile'))
    return render_template_string('''
    <h1>Welcome</h1>
    <a href="{{ url_for('register') }}">Register</a> | <a href="{{ url_for('login') }}">Login</a>
    ''')

@app.route('/register', methods=['GET', 'POST'])
def register():
    err = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not username or not password:
            err = 'Username and password are required.'
        elif get_user_by_username(username):
            err = 'Username already exists.'
        else:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                    (username, password, name, email, phone, address))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
    return render_template_string('''
    <h2>Register</h2>
    <form method="post">
        <label>Username: <input name="username"></label><br>
        <label>Password: <input type="password" name="password"></label><br>
        <label>Name: <input name="name"></label><br>
        <label>Email: <input name="email"></label><br>
        <label>Phone: <input name="phone"></label><br>
        <label>Address: <input name="address"></label><br>
        <input type="submit" value="Register">
    </form>
    <p style="color:red">{{ err }}</p>
    <a href="{{ url_for('login') }}">Login</a>
    ''', err=err)

@app.route('/login', methods=['GET', 'POST'])
def login():
    err = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = get_user_by_username(username)
        if user and user[2] == password:
            session['user_id'] = user[0]
            return redirect(url_for('profile'))
        else:
            err = 'Invalid username or password.'
    return render_template_string('''
    <h2>Login</h2>
    <form method="post">
        <label>Username: <input name="username"></label><br>
        <label>Password: <input type="password" name="password"></label><br>
        <input type="submit" value="Login">
    </form>
    <p style="color:red">{{ err }}</p>
    <a href="{{ url_for('register') }}">Register</a>
    ''', err=err)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = get_user_by_id(session['user_id'])
    msg = ''
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                  (name, email, phone, address, session['user_id']))
        conn.commit()
        conn.close()
        msg = 'Profile updated.'
        user = get_user_by_id(session['user_id'])
    return render_template_string('''
    <h2>Your Profile</h2>
    <form method="post">
        <label>Name: <input name="name" value="{{ user[3] }}"></label><br>
        <label>Email: <input name="email" value="{{ user[4] }}"></label><br>
        <label>Phone: <input name="phone" value="{{ user[5] }}"></label><br>
        <label>Address: <input name="address" value="{{ user[6] }}"></label><br>
        <input type="submit" value="Update">
    </form>
    <p style="color:green">{{ msg }}</p>
    <p><b>Account ID:</b> {{ user[0] }}</p>
    <a href="{{ url_for('logout') }}">Logout</a> | <a href="{{ url_for('view_profile_by_id', user_id=user[0]) }}">View profile by ID</a>
    ''', user=user, msg=msg)

@app.route('/profile/<int:user_id>')
def view_profile_by_id(user_id):
    user = get_user_by_id(user_id)
    if user:
        return render_template_string('''
        <h2>Profile Details</h2>
        <p><b>Account ID:</b> {{ user[0] }}</p>
        <p><b>Username:</b> {{ user[1] }}</p>
        <p><b>Name:</b> {{ user[3] }}</p>
        <p><b>Email:</b> {{ user[4] }}</p>
        <p><b>Phone:</b> {{ user[5] }}</p>
        <p><b>Address:</b> {{ user[6] }}</p>
        <a href="{{ url_for('index') }}">Home</a>
        ''', user=user)
    else:
        return render_template_string('<h2>User Not Found</h2><a href="{{ url_for(\'index\') }}">Home</a>')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)