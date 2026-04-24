import sqlite3
from flask import Flask, request, redirect, url_for, session, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-internal-portal'

def get_db_connection():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

init_db()

LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        nav { margin-bottom: 20px; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a>
        {% if 'user_id' in session %}
            {% if session.get('is_admin') %}
                | <a href="{{ url_for('admin') }}">Admin Area</a>
            {% endif %}
            | <a href="{{ url_for('logout') }}">Logout ({{ session['username'] }})</a>
        {% else %}
            | <a href="{{ url_for('login') }}">Login</a>
            | <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    <hr>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="error">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {{ content | safe }}
</body>
</html>
"""

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    content = f"<h1>Welcome, {session['username']}!</h1><p>This is the internal company portal.</p>"
    return render_template_string(LAYOUT, content=content)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        
        # Simple logic: first user becomes admin
        conn = get_db_connection()
        user_count = conn.execute('SELECT count(*) FROM users').fetchone()[0]
        is_admin = 1 if user_count == 0 else 0
        
        try:
            conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                         (username, hashed_password, is_admin))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template_string(LAYOUT, content="<h2>Register</h2><p class='error'>Username exists.</p>" + 
                '<form method="post">Username: <input name="username"><br>Password: <input type="password" name="password"><br><input type="submit"></form>')

    content = """
    <h2>Register</h2>
    <form method="post">
        Username: <input name="username" required><br><br>
        Password: <input type="password" name="password" required><br><br>
        <input type="submit" value="Sign Up">
    </form>
    """
    return render_template_string(LAYOUT, content=content)

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
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('index'))
        
        return render_template_string(LAYOUT, content="<h2>Login</h2><p class='error'>Invalid credentials.</p>" + 
            '<form method="post">Username: <input name="username"><br>Password: <input type="password" name="password"><br><input type="submit"></form>')

    content = """
    <h2>Login</h2>
    <form method="post">
        Username: <input name="username" required><br><br>
        Password: <input type="password" name="password" required><br><br>
        <input type="submit" value="Login">
    </form>
    """
    return render_template_string(LAYOUT, content=content)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return "Access Denied", 403
    
    conn = get_db_connection()
    users = conn.execute('SELECT id, username, is_admin FROM users').fetchall()
    conn.close()
    
    table_rows = ""
    for user in users:
        table_rows += f"<tr><td>{user['id']}</td><td>{user['username']}</td><td>{bool(user['is_admin'])}</td></tr>"
    
    content = f"""
    <h2>Administrator Panel - Registered Users</h2>
    <table>
        <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
        {table_rows}
    </table>
    """
    return render_template_string(LAYOUT, content=content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)