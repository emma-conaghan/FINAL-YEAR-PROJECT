import os
import anthropic
from flask import Flask, render_template_string, request, redirect, url_for, session, flash
import sqlite3
import hashlib
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

client = anthropic.Anthropic()

def get_db():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    admin_exists = conn.execute('SELECT id FROM users WHERE username = ?', ('admin',)).fetchone()
    if not admin_exists:
        admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
        conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                    ('admin', admin_password, 1))
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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
    """Generate a personalized welcome message using Claude with streaming"""
    message_content = []
    
    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": f"Generate a brief, friendly welcome message for a user named '{username}' who just logged into a company portal. Keep it to 2-3 sentences, professional but warm. Don't include any HTML tags."
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            message_content.append(text)
    
    return ''.join(message_content)

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Company Portal - {% block title %}{% endblock %}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; }
        .navbar { background: #2c3e50; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
        .navbar h1 { font-size: 1.5rem; }
        .navbar a { color: white; text-decoration: none; margin-left: 1rem; padding: 0.5rem 1rem; background: #3498db; border-radius: 4px; }
        .navbar a:hover { background: #2980b9; }
        .container { max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: bold; color: #333; }
        .form-group input { width: 100%; padding: 0.75rem; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; }
        .btn { background: #3498db; color: white; padding: 0.75rem 2rem; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; width: 100%; }
        .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .alert { padding: 1rem; border-radius: 4px; margin-bottom: 1rem; }
        .alert-error { background: #fee; border: 1px solid #e74c3c; color: #c0392b; }
        .alert-success { background: #efe; border: 1px solid #2ecc71; color: #27ae60; }
        .link { color: #3498db; text-decoration: none; }
        .link:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: bold; color: #333; }
        tr:hover { background: #f8f9fa; }
        .badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        .badge-admin { background: #e74c3c; color: white; }
        .badge-user { background: #3498db; color: white; }
        .welcome-message { font-style: italic; color: #555; margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-left: 4px solid #3498db; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="navbar">
        <h1>🏢 Company Portal</h1>
        <div>
            {% if session.get('user_id') %}
                <span style="margin-right: 1rem;">Welcome, {{ session.get('username') }}!</span>
                {% if session.get('is_admin') %}
                    <a href="/admin">Admin Panel</a>
                {% endif %}
                <a href="/logout">Logout</a>
            {% else %}
                <a href="/login">Login</a>
                <a href="/register">Register</a>
            {% endif %}
        </div>
    </div>
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

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''{% block content %}
<div class="card">
    <h2 style="margin-bottom: 1.5rem; color: #2c3e50;">🔐 Login</h2>
    <form method="POST">
        <div class="form-group">
            <label>Username</label>
            <input type="text" name="username" required placeholder="Enter your username">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" name="password" required placeholder="Enter your password">
        </div>
        <button type="submit" class="btn">Login</button>
    </form>
    <p style="margin-top: 1rem; text-align: center;">
        Don't have an account? <a href="/register" class="link">Register here</a>
    </p>
</div>
{% endblock %}''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''{% block content %}
<div class="card">
    <h2 style="margin-bottom: 1.5rem; color: #2c3e50;">📝 Register</h2>
    <form method="POST">
        <div class="form-group">
            <label>Username</label>
            <input type="text" name="username" required placeholder="Choose a username">
        </div>
        <div class="form-group">
            <label>Email (optional)</label>
            <input type="email" name="email" placeholder="Enter your email">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" name="password" required placeholder="Choose a password">
        </div>
        <div class="form-group">
            <label>Confirm Password</label>
            <input type="password" name="confirm_password" required placeholder="Confirm your password">
        </div>
        <button type="submit" class="btn">Register</button>
    </form>
    <p style="margin-top: 1rem; text-align: center;">
        Already have an account? <a href="/login" class="link">Login here</a>
    </p>
</div>
{% endblock %}''')

DASHBOARD_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''{% block content %}
<div class="card">
    <h2 style="color: #2c3e50;">👋 Welcome to the Company Portal</h2>
    {% if welcome_message %}
    <div class="welcome-message">
        {{ welcome_message }}
    </div>
    {% endif %}
    <div style="margin-top: 1.5rem; padding: 1rem; background: #f8f9fa; border-radius: 4px;">
        <h3 style="color: #2c3e50; margin-bottom: 0.5rem;">Quick Links</h3>
        <ul style="list-style: none; margin-top: 1rem;">
            <li style="margin-bottom: 0.5rem;">📊 <a href="#" class="link">Company Dashboard</a></li>
            <li style="margin-bottom: 0.5rem;">📅 <a href="#" class="link">Calendar</a></li>
            <li style="margin-bottom: 0.5rem;">📧 <a href="#" class="link">Messages</a></li>
            <li style="margin-bottom: 0.5rem;">📁 <a href="#" class="link">Documents</a></li>
        </ul>
    </div>
    {% if session.get('is_admin') %}
    <div style="margin-top: 1rem; padding: 1rem; background: #fef9e7; border-radius: 4px; border: 1px solid #f39c12;">
        <strong>🔧 Admin Notice:</strong> You have admin privileges. <a href="/admin" class="link">Access Admin Panel</a>
    </div>
    {% endif %}
</div>
{% endblock %}''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''{% block content %}
<div class="card">
    <h2 style="color: #2c3e50; margin-bottom: 1.5rem;">🔧 Admin Panel - User Management</h2>
    <p style="color: #666; margin-bottom: 1rem;">Total Users: <strong>{{ users|length }}</strong></p>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Email</th>
                <th>Role</th>
                <th>Created At</th>
            </tr>
        </thead>
        <tbody>
            {% for user in users %}
            <tr>
                <td>{{ user['id'] }}</td>
                <td>{{ user['username'] }}</td>
                <td>{{ user['email'] or 'N/A' }}</td>
                <td>
                    {% if user['is_admin'] %}
                        <span class="badge badge-admin">Admin</span>
                    {% else %}
                        <span class="badge badge-user">User</span>
                    {% endif %}
                </td>
                <td>{{ user['created_at'] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endblock %}''')

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?',
                           (username, password)).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form.get('email', '')
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        conn = get_db()
        existing_user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        
        if existing_user:
            conn.close()
            flash('Username already exists. Please choose another.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        hashed_password = hash_password(password)
        conn.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                    (username, hashed_password, email))
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/dashboard')
@login_required
def dashboard():
    welcome_message = get_ai_welcome_message(session['username'])
    return render_template_string(DASHBOARD_TEMPLATE, welcome_message=welcome_message)

@app.route('/admin')
@admin_required