from flask import Flask, request, redirect, url_for, session, render_template_string
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

def init_db():
    with sqlite3.connect('users.db') as con:
        con.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0
                      )''')
        # Create default admin if doesn't exist
        cur = con.execute("SELECT * FROM users WHERE username = ?", ('admin',))
        if cur.fetchone() is None:
            con.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                        ('admin', generate_password_hash('admin123'), 1))
        con.commit()

init_db()

register_template = '''
<h2>Register</h2>
<form method="post">
  Username: <input name="username" required><br>
  Password: <input name="password" type="password" required><br>
  <button type="submit">Register</button>
</form>
<p><a href="{{ url_for('login') }}">Login</a></p>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
'''

login_template = '''
<h2>Login</h2>
<form method="post">
  Username: <input name="username" required><br>
  Password: <input name="password" type="password" required><br>
  <button type="submit">Login</button>
</form>
<p><a href="{{ url_for('register') }}">Register</a></p>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
'''

welcome_template = '''
<h2>Welcome {{ username }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

admin_template = '''
<h2>Admin Area - Registered Users</h2>
<table border="1" cellpadding="5">
<tr><th>ID</th><th>Username</th><th>Admin</th></tr>
{% for u in users %}
<tr><td>{{ u[0] }}</td><td>{{ u[1] }}</td><td>{{ 'Yes' if u[3] else 'No' }}</td></tr>
{% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
'''

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    else:
        return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            error = "Username and password required"
        else:
            hashed = generate_password_hash(password)
            try:
                with sqlite3.connect('users.db') as con:
                    con.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                    con.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = "Username already taken"
    return render_template_string(register_template, error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        with sqlite3.connect('users.db') as con:
            cur = con.execute("SELECT id, password, is_admin FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            if row and check_password_hash(row[1], password):
                session['user_id'] = row[0]
                session['username'] = username
                session['is_admin'] = bool(row[2])
                return redirect(url_for('welcome'))
            else:
                error = "Invalid credentials"
    return render_template_string(login_template, error=error)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_template, username=session['username'], is_admin=session['is_admin'])

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    with sqlite3.connect('users.db') as con:
        cur = con.execute("SELECT * FROM users")
        users = cur.fetchall()
    return render_template_string(admin_template, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)