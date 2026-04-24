import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string

app = Flask(__name__)
app.secret_key = 'company_secret_key'

def get_db_connection():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, role TEXT)')
    conn.commit()
    conn.close()

init_db()

BASE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Company Portal</title></head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> |
        {% if 'user' in session %}
            <a href="{{ url_for('admin') }}">Admin Area</a> |
            <a href="{{ url_for('logout') }}">Logout ({{ session['user'] }})</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> |
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    <hr>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul>{% for msg in messages %}<li>{{ msg }}</li>{% endfor %}</ul>
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
"""

@app.route('/')
def index():
    if 'user' in session:
        return render_template_string(BASE_HTML + "<h1>Welcome, {{ session['user'] }}!</h1><p>This is your internal dashboard.</p>", session=session)
    return render_template_string(BASE_HTML + "<h1>Welcome to the Internal Portal</h1><p>Please login or register.</p>", session=session)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = 'admin' if username == 'admin' else 'user'
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
    return render_template_string(BASE_HTML + """
        <h2>Register</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <input type="submit" value="Sign Up">
        </form>
    """)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        conn.close()
        if user:
            session['user'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        return "Invalid credentials."
    return render_template_string(BASE_HTML + """
        <h2>Login</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <input type="submit" value="Login">
        </form>
    """)

@app.route('/admin')
def admin():
    if 'user' not in session or session.get('role') != 'admin':
        return "Access Denied: Admins Only."
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, role FROM users').fetchall()
    conn.close()
    return render_template_string(BASE_HTML + """
        <h2>Administrator Panel - User List</h2>
        <table border="1">
            <tr><th>ID</th><th>Username</th><th>Role</th></tr>
            {% for u in users %}
            <tr><td>{{ u['id'] }}</td><td>{{ u['username'] }}</td><td>{{ u['role'] }}</td></tr>
            {% endfor %}
        </table>
    """, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)