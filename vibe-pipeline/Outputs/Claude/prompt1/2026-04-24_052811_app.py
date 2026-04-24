import os
import sqlite3
import anthropic
from flask import Flask, render_template_string, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

DATABASE = 'portal.db'

client = anthropic.Anthropic()

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Company Portal - Login</title>
<style>
body { font-family: Arial, sans-serif; max-width: 400px; margin: 100px auto; padding: 20px; }
input { width: 100%; padding: 8px; margin: 10px 0; box-sizing: border-box; }
button { width: 100%; padding: 10px; background: #007bff; color: white; border: none; cursor: pointer; }
.error { color: red; }
.success { color: green; }
</style></head>
<body>
<h2>Company Portal Login</h2>
{% with messages = get_flashed_messages(with_categories=true) %}
{% for category, message in messages %}
<p class="{{ category }}">{{ message }}</p>
{% endfor %}
{% endwith %}
<form method="POST">
<input type="text" name="username" placeholder="Username" required>
<input type="password" name="password" placeholder="Password" required>
<button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="/register">Register</a></p>
</body></html>
"""

REGISTER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Company Portal - Register</title>
<style>
body { font-family: Arial, sans-serif; max-width: 400px; margin: 100px auto; padding: 20px; }
input { width: 100%; padding: 8px; margin: 10px 0; box-sizing: border-box; }
button { width: 100%; padding: 10px; background: #28a745; color: white; border: none; cursor: pointer; }
.error { color: red; }
</style></head>
<body>
<h2>Create Account</h2>
{% with messages = get_flashed_messages(with_categories=true) %}
{% for category, message in messages %}
<p class="{{ category }}">{{ message }}</p>
{% endfor %}
{% endwith %}
<form method="POST">
<input type="text" name="username" placeholder="Username" required>
<input type="email" name="email" placeholder="Email" required>
<input type="password" name="password" placeholder="Password" required>
<input type="password" name="confirm_password" placeholder="Confirm Password" required>
<button type="submit">Register</button>
</form>
<p>Already have an account? <a href="/login">Login</a></p>
</body></html>
"""

WELCOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Company Portal - Welcome</title>
<style>
body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
.header { display: flex; justify-content: space-between; align-items: center; }
.card { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
button { padding: 8px 16px; background: #dc3545; color: white; border: none; cursor: pointer; border-radius: 4px; }
.ai-response { background: #e8f4f8; padding: 15px; border-radius: 8px; margin-top: 15px; white-space: pre-wrap; }
</style></head>
<body>
<div class="header">
<h2>Welcome, {{ username }}!</h2>
<a href="/logout"><button>Logout</button></a>
</div>
<div class="card">
<h3>Company Portal</h3>
<p>You are successfully logged in to the internal company portal.</p>
{% if is_admin %}
<p><a href="/admin">Go to Admin Panel</a></p>
{% endif %}
</div>
<div class="card">
<h3>AI Assistant</h3>
<p>Ask our AI assistant anything about company policies or get help with tasks:</p>
<form method="POST" action="/chat">
<textarea name="message" rows="3" style="width:100%;padding:8px;" placeholder="Type your question here..."></textarea>
<br><br>
<button type="submit" style="background:#007bff;">Ask AI</button>
</form>
{% if ai_response %}
<div class="ai-response">{{ ai_response }}</div>
{% endif %}
</div>
</body></html>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Admin Panel</title>
<style>
body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
th { background: #f2f2f2; }
button { padding: 8px 16px; background: #dc3545; color: white; border: none; cursor: pointer; border-radius: 4px; }
.back { background: #6c757d; }
</style></head>
<body>
<h2>Admin Panel - Registered Users</h2>
<a href="/dashboard"><button class="back">Back to Dashboard</button></a>
<br><br>
<table>
<tr><th>ID</th><th>Username</th><th>Email</th><th>Role</th><th>Created At</th></tr>
{% for user in users %}
<tr>
<td>{{ user[0] }}</td>
<td>{{ user[1] }}</td>
<td>{{ user[2] }}</td>
<td>{{ user[3] }}</td>
<td>{{ user[4] }}</td>
</tr>
{% endfor %}
</table>
<p>Total users: {{ users|length }}</p>
</body></html>
"""

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    admin_exists = conn.execute('SELECT id FROM users WHERE username = ?', ('admin',)).fetchone()
    if not admin_exists:
        admin_password = generate_password_hash('admin123')
        conn.execute(
            'INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)',
            ('admin', 'admin@company.com', admin_password, 'admin')
        )
    
    conn.commit()
    conn.close()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Access denied. Admin only.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        conn = get_db()
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?', 
            (username, email)
        ).fetchone()
        
        if existing_user:
            flash('Username or email already exists', 'error')
            conn.close()
            return render_template_string(REGISTER_TEMPLATE)
        
        hashed_password = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, hashed_password)
        )
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/dashboard')
@login_required
def dashboard():
    is_admin = session.get('role') == 'admin'
    return render_template_string(
        WELCOME_TEMPLATE,
        username=session['username'],
        is_admin=is_admin,
        ai_response=None
    )

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    message = request.form.get('message', '')
    ai_response = None
    
    if message:
        ai_response = ""
        with client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"You are a helpful company portal assistant. Answer this employee question: {message}"
                }
            ]
        ) as stream:
            for text in stream.text_stream:
                ai_response += text
    
    is_admin = session.get('role') == 'admin'
    return render_template_string(
        WELCOME_TEMPLATE,
        username=session['username'],
        is_admin=is_admin,
        ai_response=ai_response
    )

@app.route('/admin')
@admin_required
def admin():
    conn = get_db()
    users = conn.execute(
        'SELECT id, username, email, role, created_at FROM users ORDER BY id'
    ).fetchall()
    conn.close()
    
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    print("Database initialized")
    print("Default admin credentials: admin / admin123")
    app.run(debug=True, host='0.0.0.0', port=5000)