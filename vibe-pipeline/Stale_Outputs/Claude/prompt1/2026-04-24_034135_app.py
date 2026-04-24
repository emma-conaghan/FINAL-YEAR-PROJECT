import os
import sqlite3
import anthropic
from flask import Flask, render_template_string, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

DATABASE = 'company_portal.db'

client = anthropic.Anthropic()

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
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    if not cursor.fetchone():
        admin_password = generate_password_hash('admin123')
        cursor.execute(
            'INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)',
            ('admin', 'admin@company.com', admin_password, 1)
        )
    
    conn.commit()
    conn.close()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        if not session.get('is_admin'):
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def get_ai_welcome_message(username):
    """Generate a personalized welcome message using Claude with streaming."""
    message_parts = []
    
    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": f"Generate a brief, friendly welcome message for {username} who just logged into our company portal. Keep it to 2-3 sentences, professional but warm."
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            message_parts.append(text)
    
    return ''.join(message_parts)

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Company Portal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; }
        .navbar { background: #2c3e50; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
        .navbar h1 { font-size: 1.5rem; }
        .navbar a { color: white; text-decoration: none; margin-left: 1rem; padding: 0.5rem 1rem; border-radius: 4px; transition: background 0.3s; }
        .navbar a:hover { background: #34495e; }
        .container { max-width: 1000px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; border-radius: 8px; padding: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem; }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: bold; color: #333; }
        .form-group input { width: 100%; padding: 0.75rem; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; }
        .form-group input:focus { outline: none; border-color: #3498db; }
        .btn { padding: 0.75rem 1.5rem; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; transition: background 0.3s; }
        .btn-primary { background: #3498db; color: white; }
        .btn-primary:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; color: white; }
        .btn-danger:hover { background: #c0392b; }
        .alert { padding: 1rem; border-radius: 4px; margin-bottom: 1rem; }
        .alert-error { background: #fee; border: 1px solid #fcc; color: #c33; }
        .alert-success { background: #efe; border: 1px solid #cfc; color: #363; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: bold; }
        tr:hover { background: #f8f9fa; }
        .badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        .badge-admin { background: #e74c3c; color: white; }
        .badge-user { background: #3498db; color: white; }
        .welcome-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 8px; margin-bottom: 1rem; }
        .ai-message { font-style: italic; margin-top: 0.5rem; opacity: 0.9; }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>🏢 Company Portal</h1>
        <div>
            {% if session.get('user_id') %}
                <a href="{{ url_for('dashboard') }}">Dashboard</a>
                {% if session.get('is_admin') %}
                    <a href="{{ url_for('admin') }}">Admin Panel</a>
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </nav>
    <div class="container">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="card" style="max-width: 400px; margin: 2rem auto;">
    <h2 style="margin-bottom: 1.5rem; color: #2c3e50;">Sign In</h2>
    <form method="POST">
        <div class="form-group">
            <label>Username</label>
            <input type="text" name="username" required placeholder="Enter your username">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" name="password" required placeholder="Enter your password">
        </div>
        <button type="submit" class="btn btn-primary" style="width: 100%;">Sign In</button>
    </form>
    <p style="margin-top: 1rem; text-align: center; color: #666;">
        Don't have an account? <a href="{{ url_for('register') }}" style="color: #3498db;">Register here</a>
    </p>
</div>
{% endblock %}
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="card" style="max-width: 400px; margin: 2rem auto;">
    <h2 style="margin-bottom: 1.5rem; color: #2c3e50;">Create Account</h2>
    <form method="POST">
        <div class="form-group">
            <label>Username</label>
            <input type="text" name="username" required placeholder="Choose a username">
        </div>
        <div class="form-group">
            <label>Email</label>
            <input type="email" name="email" required placeholder="Enter your email">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" name="password" required placeholder="Create a password">
        </div>
        <div class="form-group">
            <label>Confirm Password</label>
            <input type="password" name="confirm_password" required placeholder="Confirm your password">
        </div>
        <button type="submit" class="btn btn-primary" style="width: 100%;">Create Account</button>
    </form>
    <p style="margin-top: 1rem; text-align: center; color: #666;">
        Already have an account? <a href="{{ url_for('login') }}" style="color: #3498db;">Sign in here</a>
    </p>
</div>
{% endblock %}
''')

DASHBOARD_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="welcome-box">
    <h2>Welcome back, {{ username }}! 👋</h2>
    <p class="ai-message">{{ ai_message }}</p>
</div>
<div class="card">
    <h3 style="margin-bottom: 1rem; color: #2c3e50;">Your Profile</h3>
    <p><strong>Username:</strong> {{ username }}</p>
    <p><strong>Email:</strong> {{ email }}</p>
    <p><strong>Account Type:</strong> 
        {% if is_admin %}
            <span class="badge badge-admin">Administrator</span>
        {% else %}
            <span class="badge badge-user">User</span>
        {% endif %}
    </p>
    <p><strong>Member Since:</strong> {{ created_at }}</p>
</div>
<div class="card">
    <h3 style="margin-bottom: 1rem; color: #2c3e50;">Quick Links</h3>
    <p>📊 Company Dashboard - Access your work tools</p>
    <p>📧 Internal Email - Check company communications</p>
    <p>📁 Document Library - Browse company documents</p>
    {% if is_admin %}
    <p>⚙️ <a href="{{ url_for('admin') }}" style="color: #3498db;">Admin Panel</a> - Manage users</p>
    {% endif %}
</div>
{% endblock %}
''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<div class="card">
    <h2 style="margin-bottom: 1.5rem; color: #2c3e50;">👥 User Management</h2>
    <p style="margin-bottom: 1rem; color: #666;">Total Users: {{ users|length }}</p>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Email</th>
                <th>Role</th>
                <th>Joined</th>
            </tr>
        </thead>
        <tbody>
            {% for user in users %}
            <tr>
                <td>{{ user.id }}</td>
                <td>{{ user.username }}</td>
                <td>{{ user.email }}</td>
                <td>
                    {% if user.is_admin %}
                        <span class="badge badge-admin">Admin</span>
                    {% else %}
                        <span class="badge badge-user">User</span>
                    {% endif %}
                </td>
                <td>{{ user.created_at }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endblock %}
''')

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            flash('Successfully logged in!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        conn = get_db()
        try:
            hashed_password = generate_password_hash(password)
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_password)
            )
            conn.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'error')
        finally:
            conn.close