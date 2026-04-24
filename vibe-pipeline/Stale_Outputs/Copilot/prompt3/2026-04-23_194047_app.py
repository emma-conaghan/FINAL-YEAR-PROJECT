from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3

app = Flask(__name__)
app.secret_key = 'secret'

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, is_admin INTEGER)''')
    conn.commit()
    conn.close()

init_db()

login_template = '''
<h2>Login</h2>
<form method="post">
  Username: <input type="text" name="username"/><br/>
  Password: <input type="password" name="password"/><br/>
  <input type="submit" value="Login"/>
</form>
<p>New user? <a href="{{ url_for('register') }}">Register</a></p>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
'''

register_template = '''
<h2>Register</h2>
<form method="post">
  Username: <input type="text" name="username"/><br/>
  Password: <input type="password" name="password"/><br/>
  <input type="submit" value="Register"/>
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
'''

welcome_template = '''
<h2>Welcome {{ username }}</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

admin_template = '''
<h2>Admin - All Registered Users</h2>
<ul>
{% for user in users %}
  <li>{{ user[1] }}{% if user[3] %} (admin){% endif %}</li>
{% endfor %}
</ul>
<p><a href="{{ url_for('welcome') }}">Back</a></p>
'''

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            session['is_admin'] = user[3]
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password'
    return render_template_string(login_template, error=error)

@app.route('/register', methods=['GET','POST'])
def register():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()
        if not username or not password:
            error = 'Both fields required'
        else:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            try:
                if username == 'admin':
                    c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, 1)', (username, password))
                else:
                    c.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, 0)', (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = 'Username already taken'
            conn.close()
    return render_template_string(register_template, error=error)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_template, username=session['username'], is_admin=session.get('is_admin',0))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin',0):
        return redirect(url_for('login'))
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    users = c.fetchall()
    conn.close()
    return render_template_string(admin_template, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)