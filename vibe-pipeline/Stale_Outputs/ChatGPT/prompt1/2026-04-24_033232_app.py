from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0
        )
    ''')
    # Create default admin if not exists
    cur = conn.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cur.fetchone():
        hashed = generate_password_hash('admin')
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     ('admin', hashed, 1))
    conn.commit()
    conn.close()

init_db()

layout = '''
<!doctype html>
<title>{{ title }}</title>
<h1>{{ title }}</h1>
{% if session.get('username') %}
<p>Logged in as {{ session['username'] }} | <a href="{{ url_for('logout') }}">Logout</a></p>
{% endif %}
<hr>
{% block body %}{% endblock %}
'''

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            error = 'Username and password required.'
        else:
            conn = get_db()
            try:
                hashed = generate_password_hash(password)
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, hashed))
                conn.commit()
                conn.close()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = 'Username already taken.'
            conn.close()
    return render_template_string(layout + '''
    {% block body %}
    <form method="post">
      <label>Username: <input type="text" name="username"></label><br>
      <label>Password: <input type="password" name="password"></label><br>
      <input type="submit" value="Register">
    </form>
    {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
    <p>Already have an account? <a href="{{ url_for('login') }}">Log in</a></p>
    {% endblock %}
    ''', title="Register", error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('welcome'))
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        conn = get_db()
        cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            session['is_admin'] = user['is_admin'] == 1
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'
    return render_template_string(layout + '''
    {% block body %}
    <form method="post">
      <label>Username: <input type="text" name="username"></label><br>
      <label>Password: <input type="password" name="password"></label><br>
      <input type="submit" value="Log in">
    </form>
    {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
    <p>Don't have an account? <a href="{{ url_for('register') }}">Register</a></p>
    {% endblock %}
    ''', title="Login", error=error)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(layout + '''
    {% block body %}
    <p>Welcome, {{ session['username'] }}!</p>
    {% if session.get('is_admin') %}
    <p><a href="{{ url_for('admin') }}">Admin Area</a></p>
    {% endif %}
    {% endblock %}
    ''', title="Welcome")

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    conn = get_db()
    cur = conn.execute("SELECT username, is_admin FROM users ORDER BY username")
    users = cur.fetchall()
    conn.close()
    return render_template_string(layout + '''
    {% block body %}
    <h2>Registered Users</h2>
    <table border="1" cellpadding="5">
        <tr><th>Username</th><th>Admin</th></tr>
        {% for user in users %}
        <tr>
          <td>{{ user['username'] }}</td>
          <td>{{ 'Yes' if user['is_admin'] == 1 else 'No' }}</td>
        </tr>
        {% endfor %}
    </table>
    <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
    {% endblock %}
    ''', title="Admin Area", users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)