import sqlite3
import os
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DATABASE = 'portal.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )''')
    cursor = conn.execute("SELECT * FROM users WHERE username = 'admin'")
    if cursor.fetchone() is None:
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     ('admin', 'admin', 1))
    conn.commit()
    conn.close()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
        .navbar { width: 100%%; background: #2c3e50; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .navbar .brand { color: white; font-size: 1.3em; font-weight: bold; }
        .container { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 50px; width: 400px; max-width: 90%%; }
        .container.wide { width: 700px; }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; text-align: center; }
        form { display: flex; flex-direction: column; }
        label { margin-bottom: 5px; color: #555; font-weight: bold; }
        input[type="text"], input[type="password"] { padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 1em; }
        button, .btn { padding: 12px; background: #3498db; color: white; border: none; border-radius: 4px; font-size: 1em; cursor: pointer; text-align: center; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .message { padding: 10px; margin-bottom: 15px; border-radius: 4px; text-align: center; }
        .error { background: #fdecea; color: #c0392b; border: 1px solid #f5c6cb; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .links { text-align: center; margin-top: 15px; }
        .links a { color: #3498db; text-decoration: none; }
        .links a:hover { text-decoration: underline; }
        table { width: 100%%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .welcome-info { text-align: center; margin: 20px 0; color: #555; }
    </style>
</head>
<body>
    <div class="navbar">
        <span class="brand">Company Portal</span>
        <div>
            {% if session.get('username') %}
                <a href="{{ url_for('welcome') }}">Home</a>
                {% if session.get('is_admin') %}
                    <a href="{{ url_for('admin_panel') }}">Admin</a>
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </div>
    {{ content }}
</body>
</html>
'''

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    success = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not username or not password:
            error = 'Username and password are required.'
        elif len(username) < 3:
            error = 'Username must be at least 3 characters.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                success = 'Registration successful! You can now log in.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            finally:
                conn.close()

    content = '''
    <div class="container">
        <h2>Register</h2>
        {% if error %}
            <div class="message error">{{ error }}</div>
        {% endif %}
        {% if success %}
            <div class="message success">{{ success }}</div>
        {% endif %}
        <form method="POST">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required>
            <label for="password">Password</label>
            <input type="password" id="password" name="password" required>
            <label for="confirm_password">Confirm Password</label>
            <input type="password" id="confirm_password" name="confirm_password" required>
            <button type="submit">Register</button>
        </form>
        <div class="links">
            <p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
        </div>
    </div>
    '''

    template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(template, error=error, success=success, session=session)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                            (username, password)).fetchone()
        conn.close()

        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            session['is_admin'] = bool(user['is_admin'])
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'

    content = '''
    <div class="container">
        <h2>Login</h2>
        {% if error %}
            <div class="message error">{{ error }}</div>
        {% endif %}
        <form method="POST">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required>
            <label for="password">Password</label>
            <input type="password" id="password" name="password" required>
            <button type="submit">Login</button>
        </form>
        <div class="links">
            <p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a></p>
        </div>
    </div>
    '''

    template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(template, error=error, session=session)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        return redirect(url_for('login'))

    content = '''
    <div class="container">
        <h1>Welcome, {{ session['username'] }}!</h1>
        <div class="welcome-info">
            <p>You are logged in to the Company Portal.</p>
            <br>
            <p>This is your dashboard. Use the navigation bar to access different areas.</p>
            {% if session.get('is_admin') %}
                <br>
                <p><strong>You have administrator privileges.</strong></p>
                <br>
                <a href="{{ url_for('admin_panel') }}" class="btn">Go to Admin Panel</a>
            {% endif %}
        </div>
    </div>
    '''

    template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(template, session=session)

@app.route('/admin')
def admin_panel():
    if 'username' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        content = '''
        <div class="container">
            <h2>Access Denied</h2>
            <div class="message error">You do not have permission to access the admin area.</div>
            <div class="links">
                <a href="{{ url_for('welcome') }}">Back to Home</a>
            </div>
        </div>
        '''
        template = BASE_TEMPLATE.replace('{{ content }}', content)
        return render_template_string(template, session=session)

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()

    content = '''
    <div class="container wide">
        <h2>Admin Panel - Registered Users</h2>
        <p style="text-align:center; color:#555; margin-bottom:10px;">Total users: {{ users|length }}</p>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Username</th>
                    <th>Role</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {% for user in users %}
                <tr>
                    <td>{{ user['id'] }}</td>
                    <td>{{ user['username'] }}</td>
                    <td>{{ 'Admin' if user['is_admin'] else 'User' }}</td>
                    <td>
                        {% if not user['is_admin'] %}
                            <a href="{{ url_for('delete_user', user_id=user['id']) }}" class="btn btn-danger" style="padding:5px 10px; font-size:0.85em;" onclick="return confirm('Are you sure you want to delete this user?');">Delete</a>
                        {% else %}
                            -
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    '''

    template = BASE_TEMPLATE.replace('{{ content }}', content)
    return render_template_string(template, users=users, session=session)

@app.route('/admin/delete/<int:user_id>')
def delete_user(user_id):
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user and not user['is_admin']:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    conn.close()
    return redirect(url_for('admin_panel'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)