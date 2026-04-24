import os
import sqlite3
from flask import Flask, render_template_string, request, redirect, url_for, session, flash
from anthropic import Anthropic

app = Flask(__name__)
app.secret_key = os.urandom(24)

client = Anthropic()

DATABASE = 'portal.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        existing_admin = conn.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()
        if not existing_admin:
            conn.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)',
                        ('admin', 'admin123', 1))
        conn.commit()

BASE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .nav { background: #333; padding: 10px; margin-bottom: 20px; border-radius: 5px; }
        .nav a { color: white; text-decoration: none; margin-right: 15px; }
        .nav a:hover { text-decoration: underline; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #0056b3; }
        .btn-danger { background: #dc3545; }
        .btn-danger:hover { background: #c82333; }
        input { width: 100%; padding: 8px; margin: 5px 0 15px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        .error { color: red; }
        .success { color: green; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f2f2f2; }
        .chat-box { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
        .user-msg { background: #007bff; color: white; padding: 8px; border-radius: 5px; margin: 5px 0; text-align: right; }
        .assistant-msg { background: #f1f1f1; padding: 8px; border-radius: 5px; margin: 5px 0; }
        .chat-input { display: flex; gap: 10px; }
        .chat-input input { margin: 0; }
        .flash-msg { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    {% if session.get('user_id') %}
    <div class="nav">
        <a href="/">Home</a>
        <a href="/chat">AI Assistant</a>
        {% if session.get('is_admin') %}
        <a href="/admin">Admin Panel</a>
        {% endif %}
        <a href="/logout">Logout</a>
        <span style="color: white; float: right;">Welcome, {{ session.get('username') }}!</span>
    </div>
    {% endif %}
    
    {% with messages = get_flashed_messages(with_categories=true) %}
    {% if messages %}
    {% for category, message in messages %}
    <div class="flash-msg flash-{{ category }}">{{ message }}</div>
    {% endfor %}
    {% endif %}
    {% endwith %}
    
    {% block content %}{% endblock %}
</body>
</html>
'''

LOGIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>Login to Company Portal</h2>
<form method="POST">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Password:</label>
    <input type="password" name="password" required>
    <button type="submit" class="btn">Login</button>
</form>
<p>Don't have an account? <a href="/register">Register here</a></p>
{% endblock %}
''')

REGISTER_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>Register for Company Portal</h2>
<form method="POST">
    <label>Username:</label>
    <input type="text" name="username" required>
    <label>Password:</label>
    <input type="password" name="password" required>
    <label>Confirm Password:</label>
    <input type="password" name="confirm_password" required>
    <button type="submit" class="btn">Register</button>
</form>
<p>Already have an account? <a href="/login">Login here</a></p>
{% endblock %}
''')

HOME_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>Welcome to the Company Portal!</h2>
<p>Hello, <strong>{{ username }}</strong>! You are successfully logged in.</p>
<div style="background: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 20px;">
    <h3>Quick Links</h3>
    <ul>
        <li><a href="/chat">Chat with AI Assistant</a></li>
        {% if is_admin %}
        <li><a href="/admin">Admin Panel - Manage Users</a></li>
        {% endif %}
    </ul>
</div>
{% endblock %}
''')

ADMIN_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>Admin Panel - All Registered Users</h2>
<table>
    <tr>
        <th>ID</th>
        <th>Username</th>
        <th>Admin</th>
        <th>Created At</th>
        <th>Actions</th>
    </tr>
    {% for user in users %}
    <tr>
        <td>{{ user['id'] }}</td>
        <td>{{ user['username'] }}</td>
        <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
        <td>{{ user['created_at'] }}</td>
        <td>
            {% if not user['is_admin'] %}
            <form method="POST" action="/admin/delete/{{ user['id'] }}" style="display:inline;">
                <button type="submit" class="btn btn-danger" onclick="return confirm('Delete this user?')">Delete</button>
            </form>
            {% endif %}
        </td>
    </tr>
    {% endfor %}
</table>
{% endblock %}
''')

CHAT_TEMPLATE = BASE_TEMPLATE.replace('{% block content %}{% endblock %}', '''
{% block content %}
<h2>AI Assistant</h2>
<div class="chat-box" id="chatBox">
    {% for msg in chat_history %}
    <div class="{{ 'user-msg' if msg['role'] == 'user' else 'assistant-msg' }}">
        <strong>{{ 'You' if msg['role'] == 'user' else 'Assistant' }}:</strong> {{ msg['content'] }}
    </div>
    {% endfor %}
</div>
<form method="POST" class="chat-input">
    <input type="text" name="message" placeholder="Type your message..." required>
    <button type="submit" class="btn">Send</button>
</form>
<form method="POST" action="/chat/clear" style="margin-top: 10px;">
    <button type="submit" class="btn btn-danger">Clear Chat</button>
</form>
<script>
    var chatBox = document.getElementById('chatBox');
    chatBox.scrollTop = chatBox.scrollHeight;
</script>
{% endblock %}
''')

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string(HOME_TEMPLATE, 
                                  username=session.get('username'),
                                  is_admin=session.get('is_admin'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        with get_db() as conn:
            user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?',
                               (username, password)).fetchone()
        
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = bool(user['is_admin'])
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password:
            flash('Username and password are required', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
        else:
            try:
                with get_db() as conn:
                    conn.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                                (username, password))
                    conn.commit()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already exists', 'error')
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        flash('Access denied. Admin only.', 'error')
        return redirect(url_for('index'))
    
    with get_db() as conn:
        users = conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/admin/delete/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if not session.get('is_admin'):
        flash('Access denied. Admin only.', 'error')
        return redirect(url_for('index'))
    
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if user and not user['is_admin']:
            conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.execute('DELETE FROM chat_history WHERE user_id = ?', (user_id,))
            conn.commit()
            flash(f'User {user["username"]} deleted successfully', 'success')
        else:
            flash('Cannot delete admin users', 'error')
    
    return redirect(url_for('admin'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        
        if message:
            with get_db() as conn:
                conn.execute('INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)',
                            (user_id, 'user', message))
                conn.commit()
                
                history = conn.execute(
                    'SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY created_at',
                    (user_id,)
                ).fetchall()
            
            messages = [{"role": msg['role'], "content": msg['content']} for msg in history]
            
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=8096,
                system="You are a helpful AI assistant for a company portal. Be professional and helpful.",
                messages=messages
            )
            
            assistant_message = response.content[0].text
            
            with get_db() as conn:
                conn.execute('INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)',
                            (user_id, 'assistant', assistant_message))
                conn.commit()
    
    with get_db() as conn:
        chat_history = conn.execute(
            'SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY created_at',
            (user_id,)
        ).fetchall()
    
    return render_template_string(CHAT_TEMPLATE, chat_history=chat_history)

@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    with get_db() as conn:
        conn.execute('DELETE FROM chat_history WHERE user_id = ?', (user_id,))
        conn.commit()
    
    flash('Chat history cleared', 'success')
    return redirect(url_for('chat'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)