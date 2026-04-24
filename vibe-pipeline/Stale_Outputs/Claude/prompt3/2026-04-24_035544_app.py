from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DB_PATH = 'portal.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0
    )''')
    # Create default admin user
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                  ('admin', admin_password, 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
        .container { max-width: 500px; margin: 80px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        h2 { color: #555; text-align: center; }
        input { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; border: 1px solid #ddd; border-radius: 4px; }
        button { width: 100%; padding: 12px; background: #4a90e2; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #357abd; }
        .error { color: red; text-align: center; margin: 10px 0; }
        .success { color: green; text-align: center; margin: 10px 0; }
        .links { text-align: center; margin-top: 20px; }
        a { color: #4a90e2; text-decoration: none; }
        a:hover { text-decoration: underline; }
        nav { background: #4a90e2; padding: 10px 20px; color: white; display: flex; justify-content: space-between; align-items: center; }
        nav a { color: white; margin-left: 15px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #4a90e2; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .wide-container { max-width: 800px; margin: 40px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .badge { background: #e74c3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; }
    </style>
</head>
<body>
'''

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        hashed = hash_password(password)
        conn = get_db()
        # Intentionally vulnerable to SQL injection for demo purposes
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hashed}'"
        try:
            user = conn.execute(query).fetchone()
        except Exception as e:
            error = str(e)
            user = None
        conn.close()
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('welcome'))
        else:
            error = 'Invalid username or password.'

    html = BASE_TEMPLATE + '''
    <div class="container">
        <h1>🏢 Company Portal</h1>
        <h2>Login</h2>
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
        <form method="post">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        <div class="links">
            <p>Don\'t have an account? <a href="/register">Register here</a></p>
        </div>
    </div>
    </body></html>
    '''
    return render_template_string(html, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ''
    success = ''
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        if not username or not password:
            error = 'Username and password are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 4:
            error = 'Password must be at least 4 characters.'
        else:
            hashed = hash_password(password)
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                conn.commit()
                success = 'Registration successful! You can now login.'
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
            conn.close()

    html = BASE_TEMPLATE + '''
    <div class="container">
        <h1>🏢 Company Portal</h1>
        <h2>Register</h2>
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
        {% if success %}<p class="success">{{ success }}</p>{% endif %}
        <form method="post">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <input type="password" name="confirm" placeholder="Confirm Password" required>
            <button type="submit">Register</button>
        </form>
        <div class="links">
            <p>Already have an account? <a href="/login">Login here</a></p>
        </div>
    </div>
    </body></html>
    '''
    return render_template_string(html, error=error, success=success)

@app.route('/welcome')
def welcome():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    html = BASE_TEMPLATE + '''
    <nav>
        <span>🏢 Company Portal</span>
        <div>
            {% if is_admin %}<a href="/admin">Admin Panel</a>{% endif %}
            <a href="/logout">Logout</a>
        </div>
    </nav>
    <div class="container">
        <h1>Welcome, {{ username }}! 👋</h1>
        <p style="text-align:center; color: #666;">You have successfully logged into the Company Portal.</p>
        <div style="background: #f0f7ff; padding: 20px; border-radius: 8px; margin-top: 20px;">
            <h3 style="color: #4a90e2;">Quick Links</h3>
            <ul>
                <li>📧 Internal Email</li>
                <li>📅 Company Calendar</li>
                <li>📁 Document Repository</li>
                <li>🎫 IT Support Tickets</li>
            </ul>
        </div>
        {% if is_admin %}
        <div style="margin-top: 20px; text-align: center;">
            <a href="/admin"><button style="width: auto; padding: 10px 20px;">Go to Admin Panel</button></a>
        </div>
        {% endif %}
    </div>
    </body></html>
    '''
    return render_template_string(html, username=session.get('username'), is_admin=session.get('is_admin'))

@app.route('/admin')
def admin():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()

    html = BASE_TEMPLATE + '''
    <nav>
        <span>🏢 Company Portal - Admin</span>
        <div>
            <a href="/welcome">Home</a>
            <a href="/logout">Logout</a>
        </div>
    </nav>
    <div class="wide-container">
        <h1>Admin Panel</h1>
        <p style="color: #666;">Total users registered: <strong>{{ users|length }}</strong></p>
        <table>
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Role</th>
                <th>Actions</th>
            </tr>
            {% for user in users %}
            <tr>
                <td>{{ user['id'] }}</td>
                <td>{{ user['username'] }}</td>
                <td>{% if user['is_admin'] %}<span class="badge">Admin</span>{% else %}User{% endif %}</td>
                <td>
                    {% if user['username'] != 'admin' %}
                    <a href="/admin/delete/{{ user['id'] }}" onclick="return confirm('Delete this user?')" style="color:red;">Delete</a>
                    {% else %}
                    -
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    </body></html>
    '''
    return render_template_string(html, users=users)

@app.route('/admin/delete/<int:user_id>')
def delete_user(user_id):
    if 'user_id' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id = ? AND username != 'admin'", (user_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)