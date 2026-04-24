import os
import sqlite3
import hashlib
from anthropic import Anthropic
from flask import Flask, render_template_string, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = "super-secret-key-change-in-production"

client = Anthropic()

DB_PATH = "portal.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        admin_password = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                  ("admin", admin_password, 1))
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
    c.execute("SELECT id, username, is_admin, created_at FROM users")
    users = c.fetchall()
    conn.close()
    return users

def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_chat_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY created_at", (user_id,))
    history = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in history]

def save_message(user_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
              (user_id, role, content))
    conn.commit()
    conn.close()

def clear_chat_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
    .card { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
    button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
    button:hover { background: #0056b3; }
    .error { color: red; margin: 10px 0; }
    .success { color: green; margin: 10px 0; }
    a { color: #007bff; text-decoration: none; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 15px; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #f8f9fa; }
    .chat-container { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 4px; background: #fafafa; }
    .message { margin: 10px 0; padding: 10px; border-radius: 4px; }
    .user-message { background: #007bff; color: white; text-align: right; }
    .assistant-message { background: #e9ecef; color: #333; }
    .chat-input-area { display: flex; gap: 10px; }
    .chat-input-area input { flex: 1; }
    .chat-input-area button { width: auto; }
</style>
"""

LOGIN_TEMPLATE = f"""
<!DOCTYPE html>
<html>
<head><title>Login - Company Portal</title>{BASE_STYLE}</head>
<body>
    <div class="card">
        <h2>Company Portal Login</h2>
        {{% with messages = get_flashed_messages(with_categories=true) %}}
            {{% if messages %}}
                {{% for category, message in messages %}}
                    <p class="{{{{ 'error' if category == 'error' else 'success' }}}}">{{{{ message }}}}</p>
                {{% endfor %}}
            {{% endif %}}
        {{% endwith %}}
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        <p style="text-align: center; margin-top: 15px;">Don't have an account? <a href="/register">Register here</a></p>
    </div>
</body>
</html>
"""

REGISTER_TEMPLATE = f"""
<!DOCTYPE html>
<html>
<head><title>Register - Company Portal</title>{BASE_STYLE}</head>
<body>
    <div class="card">
        <h2>Create Account</h2>
        {{% with messages = get_flashed_messages(with_categories=true) %}}
            {{% if messages %}}
                {{% for category, message in messages %}}
                    <p class="{{{{ 'error' if category == 'error' else 'success' }}}}">{{{{ message }}}}</p>
                {{% endfor %}}
            {{% endif %}}
        {{% endwith %}}
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <input type="password" name="confirm_password" placeholder="Confirm Password" required>
            <button type="submit">Register</button>
        </form>
        <p style="text-align: center; margin-top: 15px;">Already have an account? <a href="/">Login here</a></p>
    </div>
</body>
</html>
"""

WELCOME_TEMPLATE = f"""
<!DOCTYPE html>
<html>
<head><title>Welcome - Company Portal</title>{BASE_STYLE}</head>
<body>
    <div class="card">
        <nav>
            <a href="/welcome">Home</a>
            <a href="/chat">AI Assistant</a>
            {{% if session.get('is_admin') %}}<a href="/admin">Admin Panel</a>{{% endif %}}
            <a href="/logout" style="float: right;">Logout</a>
        </nav>
        <h2>Welcome, {{{{ session['username'] }}}}!</h2>
        <p>You have successfully logged in to the Company Portal.</p>
        <p>This is your internal company portal where you can access company resources.</p>
        <p><a href="/chat">Try our AI Assistant</a> to get help with your work!</p>
    </div>
</body>
</html>
"""

CHAT_TEMPLATE = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Assistant - Company Portal</title>
    {BASE_STYLE}
    <script>
        function scrollToBottom() {{
            var container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        }}
        window.onload = scrollToBottom;
    </script>
</head>
<body>
    <div class="card">
        <nav>
            <a href="/welcome">Home</a>
            <a href="/chat">AI Assistant</a>
            {{% if session.get('is_admin') %}}<a href="/admin">Admin Panel</a>{{% endif %}}
            <a href="/logout" style="float: right;">Logout</a>
        </nav>
        <h2>AI Assistant</h2>
        <div class="chat-container" id="chat-container">
            {{% for message in chat_history %}}
                <div class="message {{{{ 'user-message' if message.role == 'user' else 'assistant-message' }}}}">
                    <strong>{{{{ 'You' if message.role == 'user' else 'Assistant' }}}}:</strong> {{{{ message.content }}}}
                </div>
            {{% endfor %}}
        </div>
        <form method="POST" action="/chat">
            <div class="chat-input-area">
                <input type="text" name="message" placeholder="Ask me anything..." required>
                <button type="submit">Send</button>
            </div>
        </form>
        <form method="POST" action="/chat/clear" style="margin-top: 10px;">
            <button type="submit" style="background: #dc3545;">Clear Chat History</button>
        </form>
    </div>
</body>
</html>
"""

ADMIN_TEMPLATE = f"""
<!DOCTYPE html>
<html>
<head><title>Admin Panel - Company Portal</title>{BASE_STYLE}</head>
<body>
    <div class="card">
        <nav>
            <a href="/welcome">Home</a>
            <a href="/chat">AI Assistant</a>
            <a href="/admin">Admin Panel</a>
            <a href="/logout" style="float: right;">Logout</a>
        </nav>
        <h2>Admin Panel - All Users</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Admin</th>
                <th>Created At</th>
            </tr>
            {{% for user in users %}}
            <tr>
                <td>{{{{ user[0] }}}}</td>
                <td>{{{{ user[1] }}}}</td>
                <td>{{{{ 'Yes' if user[2] else 'No' }}}}</td>
                <td>{{{{ user[3] }}}}</td>
            </tr>
            {{% endfor %}}
        </table>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = get_user(username)
        if user and user[2] == hash_password(password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            session["is_admin"] = bool(user[3])
            return redirect(url_for("welcome"))
        else:
            flash("Invalid username or password", "error")
    return render_template_string(LOGIN_TEMPLATE)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        if password != confirm_password:
            flash("Passwords do not match", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters", "error")
        elif len(username) < 3:
            flash("Username must be at least 3 characters", "error")
        else:
            if create_user(username, password):
                flash("Account created successfully! Please login.", "success")
                return redirect(url_for("login"))
            else:
                flash("Username already exists", "error")
    return render_template_string(REGISTER_TEMPLATE)

@app.route("/welcome")
def welcome():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template_string(WELCOME_TEMPLATE)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    user_id = session["user_id"]
    
    if request.method == "POST":
        user_message = request.form["message"]
        save_message(user_id, "user", user_message)
        history = get_chat_history(user_id)
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system="You are a helpful AI assistant for a company internal portal. Be professional and helpful.",
            messages=messages
        )
        
        assistant_message = response.content[0].text
        save_message(user_id, "assistant", assistant_message)
        return redirect(url_for("chat"))
    
    chat_history = get_chat_history(user_id)
    
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content
    
    messages = [Message(msg["role"], msg["content"]) for msg in chat_history]
    return render_template_string(CHAT_TEMPLATE, chat_history=messages)

@app.route("/chat/clear", methods=["POST"])
def clear_chat():
    if "user_id" not in session:
        return redirect(url_for("login"))
    clear_chat_history(session["user_id"])
    return redirect(url_for("chat"))

@app.route("/admin")
def admin():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if not session.get("is_admin"):
        flash("Access denied. Admin only.", "error")
        return redirect(url_for("welcome"))
    users = get_all_users()
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("login"))

if __name__ == "__