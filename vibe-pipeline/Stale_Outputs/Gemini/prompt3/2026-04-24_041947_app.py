import sqlite3
import os
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'company_secret_key'
DATABASE = 'database.db'

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
                         role TEXT NOT NULL)''')
        try:
            conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                         ('admin', 'admin123', 'admin'))
        except sqlite3.IntegrityError:
            pass
        conn.commit()

init_db()

BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Internal Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .container { max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 20px; border-radius: 10px; }
        input { display: block; width: 95%; margin-bottom: 10px; padding: 8px; }
        button { padding: 10px 15px; background: #007bff; color: white; border: none; cursor: pointer; }
        .nav { margin-bottom: 20px; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/">Home</a> | 
            {% if 'user' in session %}
                <a href="/logout">Logout</a>
                {% if session['role'] == 'admin' %} | <a href="/admin">Admin Area</a>{% endif %}
            {% else %}
                <a href="/login">Login</a> | <a href="/register">Register</a>
            {% endif %}
        </div>
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    if 'user' in session:
        return render_template_string(BASE_HTML + """
            {% block content %}
            <h1>Welcome, {{ session['user'] }}!</h1>
            <p>You are logged into the internal company portal.</p>
            {% endblock %}
        """)
    return render_template_string(BASE_HTML + """
        {% block content %}
        <h1>Internal Company Portal</h1>
        <p>Please login or register to continue.</p>
        {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        try:
            db.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                       (username, password, 'user'))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            error = "Username already exists."
    
    return render_template_string(BASE_HTML + """
        {% block content %}
        <h2>Register</h2>
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
        <form method="post">
            <input name="username" placeholder="Username" required>
            <input name="password" type="password" placeholder="Password" required>
            <button type="submit">Sign Up</button>
        </form>
        {% endblock %}
    """, error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                          (username, password)).fetchone()
        if user:
            session['user'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        error = "Invalid credentials."

    return render_template_string(BASE_HTML + """
        {% block content %}
        <h2>Login</h2>
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
        <form method="post">
            <input name="username" placeholder="Username" required>
            <input name="password" type="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        {% endblock %}
    """, error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/admin')
def admin():
    if 'user' not in session or session.get('role') != 'admin':
        return "Access Denied", 403
    
    db = get_db()
    users = db.execute("SELECT id, username, role FROM users").fetchall()
    return render_template_string(BASE_HTML + """
        {% block content %}
        <h2>Admin Panel - Registered Users</h2>
        <table border="1" width="100%" style="border-collapse: collapse;">
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Role</th>
            </tr>
            {% for user in users %}
            <tr>
                <td>{{ user['id'] }}</td>
                <td>{{ user['username'] }}</td>
                <td>{{ user['role'] }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endblock %}
    """, users=users)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)