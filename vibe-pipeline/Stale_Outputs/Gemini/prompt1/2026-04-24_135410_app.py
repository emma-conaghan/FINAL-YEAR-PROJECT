import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

DATABASE = 'database.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       username TEXT UNIQUE NOT NULL, 
                       password TEXT NOT NULL, 
                       is_admin INTEGER DEFAULT 0)''')
        db.commit()

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head><title>Internal Portal</title></head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> |
        {% if 'user_id' in session %}
            <a href="{{ url_for('logout') }}">Logout</a>
            {% if session.get('is_admin') %}
                | <a href="{{ url_for('admin') }}">Admin Panel</a>
            {% endif %}
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
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
    <h1>Welcome, {{ session['username'] }}!</h1>
    <p>This is the internal company portal.</p>
    {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        is_admin = 1 if request.form.get('is_admin') else 0
        
        db = get_db()
        try:
            db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", 
                       (username, password, is_admin))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
            
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
    <h2>Register</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        Is Admin: <input type="checkbox" name="is_admin"><br>
        <input type="submit" value="Sign Up">
    </form>
    {% endblock %}
    """)

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
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('index'))
        return "Invalid credentials."

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
    if not session.get('is_admin'):
        return "Access Denied", 403
    
    db = get_db()
    users = db.execute("SELECT id, username, is_admin FROM users").fetchall()
    
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
    <h2>Admin Area - Registered Users</h2>
    <table border="1">
        <tr><th>ID</th><th>Username</th><th>Admin Status</th></tr>
        {% for user in users %}
        <tr>
            <td>{{ user.id }}</td>
            <td>{{ user.username }}</td>
            <td>{{ 'Yes' if user.is_admin else 'No' }}</td>
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
    init_db()
    app.run(debug=True)