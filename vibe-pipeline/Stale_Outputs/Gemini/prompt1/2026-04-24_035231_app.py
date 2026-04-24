import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
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
        try:
            admin_pass = generate_password_hash('admin123')
            conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', admin_pass, 1))
        except sqlite3.IntegrityError:
            pass
        conn.commit()

BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .container { max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 20px; border-radius: 10px; }
        input { display: block; width: 100%; margin-bottom: 10px; padding: 8px; box-sizing: border-box; }
        button { background: #007BFF; color: white; border: none; padding: 10px 15px; cursor: pointer; border-radius: 5px; }
        nav { margin-bottom: 20px; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <a href="{{ url_for('index') }}">Home</a> |
            {% if 'user_id' in session %}
                <a href="{{ url_for('dashboard') }}">Dashboard</a> |
                {% if session.get('is_admin') %}
                    <a href="{{ url_for('admin_area') }}">Admin Area</a> |
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a> |
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </nav>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            {% for message in messages %}
              <p class="error">{{ message }}</p>
            {% endfor %}
          {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(BASE_HTML + "{% block content %}<h1>Welcome to the Internal Portal</h1><p>Please log in or register to continue.</p>{% endblock %}")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        
        try:
            with get_db() as conn:
                conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
            
    return render_template_string(BASE_HTML + """
    {% block content %}
    <h2>Register</h2>
    <form method="post">
        <input name="username" placeholder="Username" required>
        <input name="password" type="password" placeholder="Password" required>
        <button type="submit">Sign Up</button>
    </form>
    {% endblock %}
    """)

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
        
        return "Invalid credentials."

    return render_template_string(BASE_HTML + """
    {% block content %}
    <h2>Login</h2>
    <form method="post">
        <input name="username" placeholder="Username" required>
        <input name="password" type="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    {% endblock %}
    """)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(BASE_HTML + """
    {% block content %}
    <h1>Welcome, {{ session['username'] }}!</h1>
    <p>This is your private dashboard inside the company portal.</p>
    {% endblock %}
    """)

@app.route('/admin')
def admin_area():
    if not session.get('is_admin'):
        return "Access Denied", 403
    
    db = get_db()
    users = db.execute('SELECT id, username, is_admin FROM users').fetchall()
    
    user_list = "<ul>"
    for user in users:
        user_list += f"<li>ID: {user['id']} | Username: {user['username']} | Admin: {'Yes' if user['is_admin'] else 'No'}</li>"
    user_list += "</ul>"
    
    return render_template_string(BASE_HTML + "{% block content %}<h2>Registered Users</h2>" + user_list + "{% endblock %}")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)