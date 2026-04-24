import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_key_99'

def init_db():
    conn = sqlite3.connect('portal.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Default admin account
    try:
        cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', 
                       ('admin', generate_password_hash('admin123'), 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

init_db()

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head><title>Company Portal</title></head>
<body style="font-family: sans-serif; margin: 40px;">
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        {% if 'username' in session %}
            <a href="{{ url_for('logout') }}">Logout</a>
            {% if session.get('is_admin') %} | <a href="{{ url_for('admin') }}">Admin Panel</a>{% endif %}
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> | <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    <hr>
    {% block content %}{% endblock %}
</body>
</html>
"""

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template_string(HTML_LAYOUT + """
        {% block content %}
        <h1>Welcome, {{ session['username'] }}!</h1>
        <p>This is the internal company portal home page.</p>
        {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return "Missing fields", 400
        
        conn = sqlite3.connect('portal.db')
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                         (username, generate_password_hash(password)))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
        finally:
            conn.close()
            
    return render_template_string(HTML_LAYOUT + """
        {% block content %}
        <h2>User Registration</h2>
        <form method="post">
            <p>Username: <input type="text" name="username" required></p>
            <p>Password: <input type="password" name="password" required></p>
            <button type="submit">Sign Up</button>
        </form>
        {% endblock %}
    """)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('portal.db')
        conn.row_factory = sqlite3.Row
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('index'))
        return "Invalid username or password."

    return render_template_string(HTML_LAYOUT + """
        {% block content %}
        <h2>Login</h2>
        <form method="post">
            <p>Username: <input type="text" name="username" required></p>
            <p>Password: <input type="password" name="password" required></p>
            <button type="submit">Login</button>
        </form>
        {% endblock %}
    """)

@app.route('/admin')
def admin():
    if 'username' not in session or not session.get('is_admin'):
        return "Unauthorized", 403
    
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    users = conn.execute('SELECT id, username, is_admin FROM users').fetchall()
    conn.close()
    
    return render_template_string(HTML_LAYOUT + """
        {% block content %}
        <h2>Administrator Area</h2>
        <h3>Registered Users:</h3>
        <table border="1" cellpadding="10">
            <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
            {% for user in users %}
            <tr>
                <td>{{ user['id'] }}</td>
                <td>{{ user['username'] }}</td>
                <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
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
    app.run(debug=True, port=5000)