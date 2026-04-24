import os
import anthropic
from flask import Flask, request, session, redirect, url_for, render_template_string
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = "your-secret-key-change-in-production"

client = anthropic.Anthropic()

DB_PATH = "portal.db"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  email TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, email, created_at FROM users")
    users = c.fetchall()
    conn.close()
    return users

def create_user(username, password, email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  (username, hash_password(password), email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_ai_welcome_message(username):
    with client.messages.stream(
        model="claude-opus-4-5",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"Generate a brief, friendly welcome message for a company portal user named '{username}'. Keep it to 2-3 sentences and make it professional but warm."
            }
        ]
    ) as stream:
        message = ""
        for text in stream.text_stream:
            message += text
        return message

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
        input { width: 100%; padding: 8px; margin: 5px 0 15px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .error { color: red; padding: 10px; background: #ffeeee; border-radius: 4px; margin: 10px 0; }
        .success { color: green; padding: 10px; background: #eeffee; border-radius: 4px; margin: 10px 0; }
        nav { margin-bottom: 20px; }
        nav a { margin-right: 15px; color: #007bff; text-decoration: none; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #007bff; color: white; }
        .ai-message { background: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        {% if session.get('username') %}
        <nav>
            <a href="/welcome">Home</a>
            {% if session.get('is_admin') %}<a href="/admin">Admin Panel</a>{% endif %}
            <a href="/logout">Logout</a>
        </nav>
        {% endif %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
'''

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h2>Login to Company Portal</h2>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
<form method="POST">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Password:</label>
    <input type="password" name="password" required>
    <button type="submit">Login</button>
</form>
<p>Don't have an account? <a href="/register">Register here</a></p>
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h2>Create Account</h2>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
{% if success %}<div class="success">{{ success }}</div>{% endif %}
<form method="POST">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Email:</label>
    <input type="email" name="email">
    <label>Password:</label>
    <input type="password" name="password" required>
    <label>Confirm Password:</label>
    <input type="password" name="confirm_password" required>
    <button type="submit">Register</button>
</form>
<p>Already have an account? <a href="/login">Login here</a></p>
''')

WELCOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h2>Welcome to Company Portal</h2>
<p>Hello, <strong>{{ username }}</strong>!</p>
<div class="ai-message">
    <strong>AI Assistant:</strong>
    <p id="welcome-msg">{{ welcome_message }}</p>
</div>
<p>You are now logged in to the internal company portal. Use the navigation above to explore.</p>
''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
<h2>Admin Panel - User Management</h2>
<p>Total registered users: <strong>{{ users|length }}</strong></p>
<table>
    <tr>
        <th>ID</th>
        <th>Username</th>
        <th>Email</th>
        <th>Created At</th>
    </tr>
    {% for user in users %}
    <tr>
        <td>{{ user[0] }}</td>
        <td>{{ user[1] }}</td>
        <td>{{ user[2] or 'N/A' }}</td>
        <td>{{ user[3] }}</td>
    </tr>
    {% endfor %}
</table>
''')

@app.route('/')
def index():
    if session.get('username'):
        return redirect(url_for('welcome'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['username'] = username
            session['is_admin'] = True
            return redirect(url_for('welcome'))
        
        user = get_user(username)
        if user and user[2] == hash_password(password):
            session['username'] = username
            session['is_admin'] = False
            return redirect(url_for('welcome'))
        
        return render_template_string(LOGIN_TEMPLATE, error="Invalid username or password", session=session)
    
    return render_template_string(LOGIN_TEMPLATE, error=None, session=session)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password:
            return render_template_string(REGISTER_TEMPLATE, error="Username and password are required", success=None, session=session)
        
        if password != confirm_password:
            return render_template_string(REGISTER_TEMPLATE, error="Passwords do not match", success=None, session=session)
        
        if len(password) < 6:
            return render_template_string(REGISTER_TEMPLATE, error="Password must be at least 6 characters", success=None, session=session)
        
        if create_user(username, password, email):
            return render_template_string(REGISTER_TEMPLATE, success="Account created successfully! You can now login.", error=None, session=session)
        else:
            return render_template_string(REGISTER_TEMPLATE, error="Username already exists", success=None, session=session)
    
    return render_template_string(REGISTER_TEMPLATE, error=None, success=None, session=session)

@app.route('/welcome')
def welcome():
    if not session.get('username'):
        return redirect(url_for('login'))
    
    username = session['username']
    welcome_message = get_ai_welcome_message(username)
    
    return render_template_string(WELCOME_TEMPLATE, username=username, welcome_message=welcome_message, session=session)

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return redirect(url_for('welcome'))
    
    users = get_all_users()
    return render_template_string(ADMIN_TEMPLATE, users=users, session=session)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    print("Company Portal starting...")
    print(f"Admin credentials: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    app.run(debug=True, port=5000)