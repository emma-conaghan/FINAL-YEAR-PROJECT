from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3

app = Flask(__name__)
app.secret_key = 'secret'

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 is_admin INTEGER DEFAULT 0)''')
    # Ensure admin user exists
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                  ('admin', 'admin', 1))
    conn.commit()
    conn.close()

init_db()

login_page = '''
<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
<h2>Login</h2>
<form method="post" action="{{ url_for('login') }}">
  Username: <input type="text" name="username"><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Log In">
</form>
<p>No account? <a href="{{ url_for('register') }}">Register here</a></p>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
</body>
</html>
'''

register_page = '''
<!DOCTYPE html>
<html>
<head><title>Register</title></head>
<body>
<h2>Register</h2>
<form method="post" action="{{ url_for('register') }}">
  Username: <input type="text" name="username"><br>
  Password: <input type="password" name="password"><br>
  <input type="submit" value="Register">
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login</a></p>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
</body>
</html>
'''

welcome_page = '''
<!DOCTYPE html>
<html>
<head><title>Welcome</title></head>
<body>
<h2>Welcome, {{ user }}!</h2>
<p>This is the company portal.</p>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to admin area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
</body>
</html>
'''

admin_page = '''
<!DOCTYPE html>
<html>
<head><title>Admin Area</title></head>
<body>
<h2>Admin Area</h2>
<p>All registered users:</p>
<ul>
{% for u in users %}
  <li>{{ u[1] }}{% if u[3] %} (admin){% endif %}</li>
{% endfor %}
</ul>
<p><a href="{{ url_for('welcome') }}">Back</a></p>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
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
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            session['is_admin'] = user[3]
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid credentials.'
    return render_template_string(login_page, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            error = 'Username and password required.'
        else:
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                conn.close()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = 'Username already taken.'
    return render_template_string(register_page, error=error)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(welcome_page, user=session['username'], is_admin=session.get('is_admin', 0))

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin', 0):
        return redirect(url_for('login'))
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users")
    users = c.fetchall()
    conn.close()
    return render_template_string(admin_page, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)