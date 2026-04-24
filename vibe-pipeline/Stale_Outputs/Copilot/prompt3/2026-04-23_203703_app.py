from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3

app = Flask(__name__)
app.secret_key = 'secret'


def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Create default admin user if not exists
    c.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if not c.fetchone():
        c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', 'admin', 1))
    conn.commit()
    conn.close()


init_db()

login_html = '''
<h2>Login</h2>
<form method="POST">
  Username: <input type="text" name="username"><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Login">
</form>
<p>No account? <a href="{{ url_for('register') }}">Register here</a></p>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
'''

register_html = '''
<h2>Register</h2>
<form method="POST">
  Username: <input type="text" name="username"><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Register">
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
'''

welcome_html = '''
<h2>Welcome, {{ username }}!</h2>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
{% if admin %}<p><a href="{{ url_for('admin') }}">Admin Area</a></p>{% endif %}
'''

admin_html = '''
<h2>Admin Area</h2>
<table border="1">
<tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
{% for user in users %}
<tr>
  <td>{{ user[0] }}</td>
  <td>{{ user[1] }}</td>
  <td>{{ 'Yes' if user[3] else 'No' }}</td>
</tr>
{% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            session['admin'] = user[3]
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password'
    return render_template_string(login_html, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            error = 'Please fill both fields.'
        else:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            try:
                c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            conn.close()
    return render_template_string(register_html, error=error)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_html, username=session['username'], admin=session.get('admin'))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('admin'):
        return redirect(url_for('login'))
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    users = c.fetchall()
    conn.close()
    return render_template_string(admin_html, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)