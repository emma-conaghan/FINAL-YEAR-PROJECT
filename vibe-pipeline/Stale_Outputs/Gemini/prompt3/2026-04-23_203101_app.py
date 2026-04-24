import sqlite3
from flask import Flask, request, redirect, url_for, session, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

DATABASE = 'portal.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        # Create a default admin if it doesn't exist
        try:
            conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                         ('admin', generate_password_hash('admin123'), 1))
        except sqlite3.IntegrityError:
            pass
        conn.commit()

init_db()

LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 20px; border-radius: 8px; }
        nav { margin-bottom: 20px; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <a href="/">Home</a>
            {% if 'user_id' in session %}
                | <a href="/dashboard">Dashboard</a>
                {% if session.get('is_admin') %} | <a href="/admin">Admin Area</a> {% endif %}
                | <a href="/logout">Logout</a>
            {% else %}
                | <a href="/login">Login</a> | <a href="/register">Register</a>
            {% endif %}
        </nav>
        <hr>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            {% for message in messages %} <p class="error">{{ message }}</p> {% endfor %}
          {% endif %}
        {% endwith %}
        {{ content | safe }}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    content = "<h1>Welcome to the Internal Portal</h1><p>Please log in or register to continue.</p>"
    return render_template_string(LAYOUT, content=content)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        db = get_db()
        try:
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template_string(LAYOUT, content="<h2>Register</h2><p class='error'>Username exists.</p>" + registration_form())
    return render_template_string(LAYOUT, content="<h2>Register</h2>" + registration_form())

def registration_form():
    return """
    <form method="post">
        Username: <input type="text" name="username" required><br><br>
        Password: <input type="password" name="password" required><br><br>
        <input type="submit" value="Register">
    </form>
    """

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('dashboard'))
        return render_template_string(LAYOUT, content="<h2>Login</h2><p class='error'>Invalid credentials.</p>" + login_form())
    return render_template_string(LAYOUT, content="<h2>Login</h2>" + login_form())

def login_form():
    return """
    <form method="post">
        Username: <input type="text" name="username" required><br><br>
        Password: <input type="password" name="password" required><br><br>
        <input type="submit" value="Login">
    </form>
    """

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    content = f"<h1>Welcome, {session['username']}!</h1><p>You have successfully logged into the internal portal.</p>"
    return render_template_string(LAYOUT, content=content)

@app.route('/admin')
def admin():
    if 'user_id' not in session or not session.get('is_admin'):
        return "Access Denied", 403
    db = get_db()
    users = db.execute('SELECT id, username, is_admin FROM users').fetchall()
    user_list = "".join([f"<li>ID: {u['id']} - Username: {u['username']} {'(Admin)' if u['is_admin'] else ''}</li>" for u in users])
    content = f"<h1>Administrator Panel</h1><h3>Registered Users:</h3><ul>{user_list}</ul>"
    return render_template_string(LAYOUT, content=content)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)