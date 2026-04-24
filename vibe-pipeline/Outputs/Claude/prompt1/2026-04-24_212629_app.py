import os
import sqlite3
import hashlib
from flask import Flask, render_template_string, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DATABASE = 'users.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    # Create default admin user
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password, is_admin) VALUES (?, ?, ?)
    ''', ('admin', admin_password, 1))
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
        .navbar { background: #2c3e50; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 500px; margin: 60px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h2 { color: #2c3e50; margin-bottom: 25px; text-align: center; }
        input[type=text], input[type=password] { width: 100%; padding: 10px; margin: 8px 0 16px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        button, .btn { background: #2c3e50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; width: 100%; }
        button:hover, .btn:hover { background: #34495e; }
        .alert { padding: 10px; margin-bottom: 15px; border-radius: 4px; }
        .alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .wide-container { max-width: 800px; margin: 60px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        p { color: #555; line-height: 1.6; }
        .link { text-align: center; margin-top: 15px; font-size: 13px; color: #666; }
        .link a { color: #2c3e50; text-decoration: none; font-weight: bold; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; background: #e74c3c; color: white; }
    </style>
</head>
<body>
    <div class="navbar">
        <div><strong>🏢 Company Portal</strong></div>
        <div>
            {% if session.get('username') %}
                Welcome, {{ session['username'] }}!
                {% if session.get('is_admin') %}<a href="{{ url_for('admin') }}">Admin Panel</a>{% endif %}
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </div>
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            <div style="max-width:500px;margin:20px auto;">
            {% for category, message in messages %}
                <div class="alert alert-{{ category }}">{{ message }}</div>
            {% endfor %}
            </div>
        {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
'''

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="container">
    <h2>🔐 Login</h2>
    <form method="POST">
        <label>Username</label>
        <input type="text" name="username" placeholder="Enter username" required>
        <label>Password</label>
        <input type="password" name="password" placeholder="Enter password" required>
        <button type="submit">Login</button>
    </form>
    <div class="link">Don't have an account? <a href="{{ url_for('register') }}">Register here</a></div>
</div>
{% endblock %}
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="container">
    <h2>📝 Register</h2>
    <form method="POST">
        <label>Username</label>
        <input type="text" name="username" placeholder="Choose a username" required>
        <label>Password</label>
        <input type="password" name="password" placeholder="Choose a password" required>
        <label>Confirm Password</label>
        <input type="password" name="confirm_password" placeholder="Confirm your password" required>
        <button type="submit">Register</button>
    </form>
    <div class="link">Already have an account? <a href="{{ url_for('login') }}">Login here</a></div>
</div>
{% endblock %}
''')

WELCOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="container" style="max-width:600px;">
    <h2>👋 Welcome, {{ username }}!</h2>
    <div class="alert alert-success">You have successfully logged in to the Company Portal.</div>
    <p>This is your internal company portal. Here you can access company resources, communicate with your team, and stay up to date with the latest news.</p>
    <p><strong>Quick Links:</strong></p>
    <ul style="color:#555; line-height:2;">
        <li>📅 Company Calendar</li>
        <li>📁 Document Library</li>
        <li>💬 Team Chat</li>
        <li>📊 Reports & Analytics</li>
    </ul>
    {% if is_admin %}
    <a href="{{ url_for('admin') }}" class="btn" style="text-decoration:none; display:block; text-align:center; margin-top:20px;">Go to Admin Panel</a>
    {% endif %}
</div>
{% endblock %}
''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="wide-container">
    <h2>⚙️ Admin Panel</h2>
    <p>Total registered users: <strong>{{ users|length }}</strong></p>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Username</th>
                <th>Role</th>
            </tr>
        </thead>
        <tbody>
            {% for user in users %}
            <tr>
                <td>{{ user['id'] }}</td>
                <td>{{ user['username'] }}</td>
                <td>{% if user['is_admin'] %}<span class="badge">Admin</span>{% else %}User{% endif %}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endblock %}
''')

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('welcome'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template_string(LOGIN_TEMPLATE)
        
        conn = get_db()
        cursor = conn.cursor()
        hashed = hash_password(password)
        # Intentionally vulnerable to SQL injection for demonstration (as requested: insecure is ok)
        query = "SELECT * FROM users WHERE username = '{}' AND password = '{}'".format(username, hashed)
        cursor.execute(query)
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            session['is_admin'] = bool(user['is_admin'])
            flash('Login successful!', 'success')
            return redirect(url_for('welcome'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('welcome'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if len(password) < 4:
            flash('Password must be at least 4 characters long.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        existing = cursor.fetchone()
        
        if existing:
            flash('Username already taken. Please choose another.', 'error')
            conn.close()
            return render_template_string(REGISTER_TEMPLATE)
        
        hashed = hash_password(password)
        cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', (username, hashed, 0))
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        flash('Please log in to access this page.', 'info')
        return redirect(url_for('login'))
    
    return render_template_string(
        WELCOME_TEMPLATE,
        username=session['username'],
        is_admin=session.get('is_admin', False)
    )

@app.route('/admin')
def admin():
    if 'username' not in session:
        flash('Please log in to access this page.', 'info')
        return redirect(url_for('login'))
    
    if not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('welcome'))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, is_admin FROM users ORDER BY id')
    users = cursor.fetchall()
    conn.close()
    
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)