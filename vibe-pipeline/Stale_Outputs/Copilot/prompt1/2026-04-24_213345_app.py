from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3

app = Flask(__name__)
app.secret_key = 'secret-key'

def init_db():
    conn = sqlite3.connect('portal.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )
    ''')
    # Create admin user if not exists
    c.execute('SELECT * FROM users WHERE username=?', ('admin',))
    if not c.fetchone():
        c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', 'admin', 1))
    conn.commit()
    conn.close()

init_db()

login_page = '''
<h2>Login</h2>
<form method="post">
  Username: <input name="username"><br>
  Password: <input name="password" type="password"><br>
  <input type="submit" value="Login">
</form>
<p>Don't have an account? <a href="/register">Register here</a></p>
{% if error %}<p style="color:red">{{error}}</p>{% endif %}
'''

register_page = '''
<h2>Register</h2>
<form method="post">
  Username: <input name="username"><br>
  Password: <input name="password" type="password"><br>
  <input type="submit" value="Register">
</form>
<p>Already have an account? <a href="/login">Login here</a></p>
{% if error %}<p style="color:red">{{error}}</p>{% endif %}
'''

welcome_page = '''
<h2>Welcome {{username}}!</h2>
<p><a href="/logout">Logout</a></p>
{% if is_admin %}<p><a href="/admin">Go to Admin Area</a></p>{% endif %}
'''

admin_page = '''
<h2>Admin Area - Registered Users</h2>
<ul>
{% for user in users %}
  <li>{{user[1]}}{% if user[3] %} (admin){% endif %}</li>
{% endfor %}
</ul>
<p><a href="/">Back to Welcome</a></p>
'''

@app.route('/')
def index():
    if 'username' not in session:
        return redirect('/login')
    is_admin = session.get('is_admin', False)
    return render_template_string(welcome_page, username=session['username'], is_admin=is_admin)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('portal.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = user[1]
            session['is_admin'] = user[3]
            return redirect('/')
        else:
            error = "Invalid credentials."
    return render_template_string(login_page, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            error = "Please fill out all fields."
        else:
            try:
                conn = sqlite3.connect('portal.db')
                c = conn.cursor()
                c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
                conn.commit()
                conn.close()
                return redirect('/login')
            except sqlite3.IntegrityError:
                error = "Username already exists."
    return render_template_string(register_page, error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect('/')
    conn = sqlite3.connect('portal.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    users = c.fetchall()
    conn.close()
    return render_template_string(admin_page, users=users)

if __name__ == '__main__':
    app.run(debug=True)