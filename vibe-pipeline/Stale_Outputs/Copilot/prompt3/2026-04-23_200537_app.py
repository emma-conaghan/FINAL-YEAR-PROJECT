from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3

app = Flask(__name__)
app.secret_key = 'your-secret-key'

db_path = 'users.db'

def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE,
                 password TEXT,
                 is_admin INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

init_db()

login_template = '''
<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
<h2>Login</h2>
<form method="post">
  Username: <input type="text" name="username" required><br>
  Password: <input type="password" name="password" required><br>
  <input type="submit" value="Login">
</form>
<a href="{{ url_for('register') }}">Register</a>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
</body>
</html>
'''

register_template = '''
<!DOCTYPE html>
<html>
<head><title>Register</title></head>
<body>
<h2>Register</h2>
<form method="post">
  Username: <input type="text" name="username" required><br>
  Password: <input type="password" name="password" required><br>
  <input type="submit" value="Register">
</form>
<a href="{{ url_for('login') }}">Login</a>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
</body>
</html>
'''

welcome_template = '''
<!DOCTYPE html>
<html>
<head><title>Welcome</title></head>
<body>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
</body>
</html>
'''

admin_template = '''
<!DOCTYPE html>
<html>
<head><title>Admin Area</title></head>
<body>
<h2>Admin Area</h2>
<p>All Registered Users:</p>
<ul>
{% for user in users %}
  <li>{{ user[1] }} {% if user[3]==1 %}(admin){% endif %}</li>
{% endfor %}
</ul>
<a href="{{ url_for('welcome') }}">Back</a>
</body>
</html>
'''

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            session['is_admin'] = bool(user[3])
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
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        try:
            # First user becomes admin
            c.execute('SELECT COUNT(*) FROM users')
            is_admin = 1 if c.fetchone()[0] == 0 else 0
            c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', (username, password, is_admin))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = 'Username already taken'
            conn.close()
    return render_template_string(register_template, error=error)

@app.route('/welcome')
def welcome():
    if 'username' in session:
        return render_template_string(welcome_template, username=session['username'], is_admin=session.get('is_admin', False))
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('is_admin', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if 'username' in session and session.get('is_admin'):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM users')
        users = c.fetchall()
        conn.close()
        return render_template_string(admin_template, users=users)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)