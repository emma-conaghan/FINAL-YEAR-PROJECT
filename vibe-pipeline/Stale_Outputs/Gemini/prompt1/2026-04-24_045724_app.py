import sqlite3
from flask import Flask, request, render_template_string, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "temporary_secret_key_for_internal_use"

DATABASE = 'portal.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                         username TEXT UNIQUE NOT NULL, 
                         password TEXT NOT NULL,
                         is_admin INTEGER DEFAULT 0)''')
        conn.commit()

init_db()

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head><title>Company Portal</title></head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        {% if 'user_id' in session %}
            <a href="{{ url_for('admin') }}">Admin Area</a> | 
            <a href="{{ url_for('logout') }}">Logout</a>
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
    {{ content | safe }}
</body>
</html>
"""

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    content = f"<h1>Welcome, {session['username']}!</h1><p>This is the internal company portal.</p>"
    return render_template_string(HTML_LAYOUT, content=content)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        is_admin = 1 if username.lower() == 'admin' else 0
        try:
            db = get_db()
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", (username, password, is_admin))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template_string(HTML_LAYOUT, content="<h1>Register</h1><p>Username already exists.</p>")
    
    content = """
    <h1>Register</h1>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Sign Up">
    </form>
    """
    return render_template_string(HTML_LAYOUT, content=content)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('index'))
        return render_template_string(HTML_LAYOUT, content="<h1>Login</h1><p>Invalid credentials.</p>")
    
    content = """
    <h1>Login</h1>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Login">
    </form>
    """
    return render_template_string(HTML_LAYOUT, content=content)

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return "Access Denied: Admins Only", 403
    
    db = get_db()
    users = db.execute("SELECT id, username, is_admin FROM users").fetchall()
    
    user_list = "<ul>"
    for user in users:
        role = "Admin" if user['is_admin'] else "User"
        user_list += f"<li>ID: {user['id']} - Username: {user['username']} ({role})</li>"
    user_list += "</ul>"
    
    content = f"<h1>Admin Dashboard</h1><h3>Registered Users:</h3>{user_list}"
    return render_template_string(HTML_LAYOUT, content=content)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)