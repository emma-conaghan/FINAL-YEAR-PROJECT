import sqlite3
from flask import Flask, request, redirect, url_for, session, render_template_string, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_secret_key_12345'
DATABASE = 'portal.db'

def query_db(query, args=(), one=False):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, args)
    rv = cur.fetchall()
    conn.commit()
    conn.close()
    return (rv[0] if rv else None) if one else rv

def init_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     username TEXT UNIQUE NOT NULL, 
                     password TEXT NOT NULL, 
                     is_admin INTEGER DEFAULT 0)''')
    try:
        conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', 
                     ('admin', generate_password_hash('admin123'), 1))
    except sqlite3.IntegrityError:
        pass
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
        nav { margin-bottom: 20px; padding: 10px; background: #eee; }
        .error { color: red; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        {% if session.get('user_id') %}
            {% if session.get('is_admin') %}<a href="{{ url_for('admin') }}">Admin Area</a> | {% endif %}
            <a href="{{ url_for('logout') }}">Logout ({{ session.get('username') }})</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> | <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    {% with messages = get_flashed_messages() %}
        {% if messages %}
            {% for message in messages %}<p class="error">{{ message }}</p>{% endfor %}
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
    return render_template_string(LAYOUT + """
        {% block content %}
        <h1>Welcome to the Internal Portal</h1>
        <p>Hello, <strong>{{ session['username'] }}</strong>. You are successfully authenticated.</p>
        {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash("All fields required.")
        else:
            hashed_pw = generate_password_hash(password)
            try:
                query_db('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
                flash("Registration successful. Please log in.")
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash("Username already exists.")
    return render_template_string(LAYOUT + """
        {% block content %}
        <h2>User Registration</h2>
        <form method="post">
            Username: <input type="text" name="username"><br><br>
            Password: <input type="password" name="password"><br><br>
            <input type="submit" value="Register">
        </form>
        {% endblock %}
    """)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = query_db('SELECT * FROM users WHERE username = ?', [username], one=True)
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.")
    return render_template_string(LAYOUT + """
        {% block content %}
        <h2>Login</h2>
        <form method="post">
            Username: <input type="text" name="username"><br><br>
            Password: <input type="password" name="password"><br><br>
            <input type="submit" value="Login">
        </form>
        {% endblock %}
    """)

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return "Access Denied: Administrators only.", 403
    users = query_db('SELECT id, username, is_admin FROM users')
    return render_template_string(LAYOUT + """
        {% block content %}
        <h2>Admin Area - Registered Users</h2>
        <table border="1" cellpadding="10">
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Role</th>
            </tr>
            {% for user in users %}
            <tr>
                <td>{{ user['id'] }}</td>
                <td>{{ user['username'] }}</td>
                <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endblock %}
    """, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)