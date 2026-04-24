import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret'

def get_db_connection():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, is_admin INTEGER DEFAULT 0)')
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
            <a href="{{ url_for('admin') }}">Admin</a> | 
            <a href="{{ url_for('logout') }}">Logout ({{ session['username'] }})</a>
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
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
        <h1>Welcome, {{ session['username'] }}!</h1>
        <p>This is the internal company portal dashboard.</p>
    {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists"
            
    return render_template_string(HTML_LAYOUT + """
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
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
        return "Invalid credentials"

    return render_template_string(HTML_LAYOUT + """
    {% block content %}
        <h2>Login</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <input type="submit" value="Login">
        </form>
    {% endblock %}
    """)

@app.route('/admin')
def admin():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    users = conn.execute('SELECT id, username FROM users').fetchall()
    conn.close()
    
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
        <h2>Admin Area - Registered Users</h2>
        <ul>
            {% for user in users %}
                <li>ID: {{ user['id'] }} - Username: {{ user['username'] }}</li>
            {% endfor %}
        </ul>
    {% endblock %}
    """, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)