import sqlite3
import os
from flask import Flask, request, render_template_string, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.urandom(24)
DB_PATH = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password TEXT, 
                  is_admin INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

init_db()

LAYOUT = """
<!DOCTYPE html>
<html>
<head><title>Company Portal</title></head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        {% if 'user' in session %}
            <a href="{{ url_for('welcome') }}">Dashboard</a> | 
            <a href="{{ url_for('admin') }}">Admin</a> | 
            <a href="{{ url_for('logout') }}">Logout ({{ session['user'] }})</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> | 
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    <hr>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p style="color: red;">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(LAYOUT + "{% block content %}<h1>Welcome to the Internal Portal</h1><p>Please log in to access the dashboard.</p>{% endblock %}")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
    
    return render_template_string(LAYOUT + """
    {% block content %}
    <h2>Register</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Sign Up">
    </form>
    {% endblock %}
    """)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[0], password):
            session['user'] = username
            return redirect(url_for('welcome'))
        return "Invalid credentials"

    return render_template_string(LAYOUT + """
    {% block content %}
    <h2>Login</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Login">
    </form>
    {% endblock %}
    """)

@app.route('/welcome')
def welcome():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string(LAYOUT + """
    {% block content %}
    <h1>Dashboard</h1>
    <p>Welcome back, {{ session['user'] }}! You have successfully authenticated.</p>
    {% endblock %}
    """)

@app.route('/admin')
def admin():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username FROM users")
    users = c.fetchall()
    conn.close()
    
    return render_template_string(LAYOUT + """
    {% block content %}
    <h2>Admin Area - Registered Users</h2>
    <table border="1">
        <tr><th>ID</th><th>Username</th></tr>
        {% for user in users %}
        <tr><td>{{ user[0] }}</td><td>{{ user[1] }}</td></tr>
        {% endfor %}
    </table>
    {% endblock %}
    """, users=users)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)