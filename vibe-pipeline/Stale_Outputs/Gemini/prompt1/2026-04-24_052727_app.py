import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string

app = Flask(__name__)
app.secret_key = "super_secret_key_for_internal_portal"

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, is_admin INTEGER DEFAULT 0)')
    # Create a default admin if not exists
    try:
        conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', 'admin123', 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

init_db()

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head><title>Company Portal</title></head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> |
        {% if 'user_id' in session %}
            <a href="{{ url_for('dashboard') }}">Dashboard</a> |
            {% if session.get('is_admin') %}
                <a href="{{ url_for('admin') }}">Admin Area</a> |
            {% endif %}
            <a href="{{ url_for('logout') }}">Logout</a>
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
    return render_template_string(HTML_LAYOUT + "<h1>Welcome to the Internal Portal</h1><p>Please log in or register to continue.</p>")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
        finally:
            conn.close()
    return render_template_string(HTML_LAYOUT + """
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
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('dashboard'))
        return "Invalid credentials."
    return render_template_string(HTML_LAYOUT + """
        <h2>Login</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <input type="submit" value="Login">
        </form>
    """)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(HTML_LAYOUT + "<h1>Dashboard</h1><p>Welcome, {{ session['username'] }}!</p>")

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return "Access Denied", 403
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, is_admin FROM users').fetchall()
    conn.close()
    return render_template_string(HTML_LAYOUT + """
        <h2>Admin Area - Registered Users</h2>
        <table border="1">
            <tr><th>ID</th><th>Username</th><th>Admin?</th></tr>
            {% for user in users %}
            <tr>
                <td>{{ user.id }}</td>
                <td>{{ user.username }}</td>
                <td>{{ 'Yes' if user.is_admin else 'No' }}</td>
            </tr>
            {% endfor %}
        </table>
    """, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)