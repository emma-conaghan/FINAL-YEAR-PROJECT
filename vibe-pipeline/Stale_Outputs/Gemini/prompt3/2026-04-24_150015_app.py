import sqlite3
from flask import Flask, request, redirect, url_for, session, render_template_string, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_portal'

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
    # Create default admin if not exists
    try:
        admin_pass = generate_password_hash('admin123')
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', admin_pass, 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

init_db()

HTML_WRAPPER = """
<!DOCTYPE html>
<html>
<head>
    <title>Internal Company Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        nav a { margin-right: 15px; text-decoration: none; color: blue; }
        .error { color: red; }
        .success { color: green; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a>
        {% if 'user_id' in session %}
            {% if session.get('is_admin') %}
                <a href="{{ url_for('admin') }}">Admin Panel</a>
            {% endif %}
            <a href="{{ url_for('logout') }}">Logout ({{ session['username'] }})</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a>
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="flash">{{ message }}</p>
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
        content = "<h1>Welcome to the Company Portal</h1><p>Please <a href='/login'>login</a> or <a href='/register'>register</a> to continue.</p>"
    else:
        content = f"<h1>Welcome, {session['username']}!</h1><p>This is the internal dashboard for company employees.</p>"
    return render_template_string(HTML_WRAPPER, content=content)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.')
        finally:
            conn.close()
            
    content = """
    <h1>Register</h1>
    <form method="post">
        Username: <br><input type="text" name="username" required><br>
        Password: <br><input type="password" name="password" required><br><br>
        <input type="submit" value="Sign Up">
    </form>
    """
    return render_template_string(HTML_WRAPPER, content=content)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.')
            
    content = """
    <h1>Login</h1>
    <form method="post">
        Username: <br><input type="text" name="username" required><br>
        Password: <br><input type="password" name="password" required><br><br>
        <input type="submit" value="Login">
    </form>
    """
    return render_template_string(HTML_WRAPPER, content=content)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        flash('Access denied. Admins only.')
        return redirect(url_for('index'))
    
    conn = get_db_connection()
    users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
    conn.close()
    
    user_rows = ""
    for u in users:
        role = "Admin" if u['is_admin'] else "User"
        user_rows += f"<tr><td>{u['id']}</td><td>{u['username']}</td><td>{role}</td></tr>"
        
    content = f"""
    <h1>Admin Panel</h1>
    <h3>Registered Users</h3>
    <table>
        <tr><th>ID</th><th>Username</th><th>Role</th></tr>
        {user_rows}
    </table>
    """
    return render_template_string(HTML_WRAPPER, content=content)

if __name__ == '__main__':
    app.run(debug=True)