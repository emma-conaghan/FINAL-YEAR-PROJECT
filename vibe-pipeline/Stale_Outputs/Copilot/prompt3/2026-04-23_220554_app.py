from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3

app = Flask(__name__)
app.secret_key = 'secret'

def init_db():
    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE,
                      password TEXT,
                      is_admin INTEGER DEFAULT 0)''')
        c.execute('SELECT * FROM users WHERE is_admin=1')
        if not c.fetchone():
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                      ('admin', 'adminpass', 1))
        conn.commit()

init_db()

login_page = '''
<html>
<head><title>Login</title></head>
<body>
<h2>Login</h2>
<form method="post">
  Username: <input type="text" name="username"><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Login">
</form>
<p><a href="/register">Register</a></p>
<p style="color:red;">{{ error }}</p>
</body>
</html>
'''

register_page = '''
<html>
<head><title>Register</title></head>
<body>
<h2>Register</h2>
<form method="post">
  Username: <input type="text" name="username"><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Register">
</form>
<p><a href="/login">Back to Login</a></p>
<p style="color:red;">{{ error }}</p>
</body>
</html>
'''

welcome_page = '''
<html>
<head><title>Welcome</title></head>
<body>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="/admin">Admin Area</a></p>
{% endif %}
<p><a href="/logout">Logout</a></p>
</body>
</html>
'''

admin_page = '''
<html>
<head><title>Admin Area</title></head>
<body>
<h2>Admin Area</h2>
<table border="1">
<tr><th>ID</th><th>Username</th><th>Admin</th></tr>
{% for user in users %}
<tr>
<td>{{ user[0] }}</td>
<td>{{ user[1] }}</td>
<td>{{ 'Yes' if user[3] else 'No' }}</td>
</tr>
{% endfor %}
</table>
<p><a href="/welcome">Back</a></p>
</body>
</html>
'''

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        with sqlite3.connect('users.db') as conn:
            c = conn.cursor()
            c.execute('SELECT id, password, is_admin FROM users WHERE username=?', (username,))
            row = c.fetchone()
            if row and row[1] == password:
                session['user_id'] = row[0]
                session['username'] = username
                session['is_admin'] = bool(row[2])
                return redirect('/welcome')
            else:
                error = 'Invalid username or password'
    return render_template_string(login_page, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            error = 'Please enter both fields'
        else:
            try:
                with sqlite3.connect('users.db') as conn:
                    c = conn.cursor()
                    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
                    conn.commit()
                return redirect('/login')
            except sqlite3.IntegrityError:
                error = 'Username already exists'
    return render_template_string(register_page, error=error)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template_string(welcome_page, username=session.get('username'), is_admin=session.get('is_admin'))

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return redirect('/login')
    with sqlite3.connect('users.db') as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM users')
        users = c.fetchall()
    return render_template_string(admin_page, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)