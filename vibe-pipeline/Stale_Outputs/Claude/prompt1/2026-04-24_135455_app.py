import os
import sqlite3
import hashlib
from flask import Flask, render_template_string, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = "supersecretkey123"

DATABASE = "users.db"

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    """)
    # Create default admin user if not exists
    admin_password = hashlib.md5("admin123".encode()).hexdigest()
    cursor.execute("INSERT OR IGNORE INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                   ("admin", admin_password, 1))
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
        .container { max-width: 500px; margin: 80px auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; text-align: center; }
        input[type=text], input[type=password] { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button, .btn { width: 100%; padding: 12px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; text-decoration: none; display: block; text-align: center; box-sizing: border-box; }
        button:hover, .btn:hover { background: #45a049; }
        .btn-red { background: #e74c3c; }
        .btn-red:hover { background: #c0392b; }
        .btn-blue { background: #3498db; }
        .btn-blue:hover { background: #2980b9; }
        .flash { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .flash.error { background: #ffe0e0; color: #c00; }
        .flash.success { background: #e0ffe0; color: #060; }
        .flash.info { background: #e0f0ff; color: #006; }
        .link { text-align: center; margin-top: 20px; }
        .link a { color: #3498db; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .nav { display: flex; gap: 10px; margin-bottom: 20px; }
        .nav a { flex: 1; }
        .wide { max-width: 800px; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; background: #e74c3c; color: white; }
    </style>
</head>
<body>
    <div class="container {% if wide %}wide{% endif %}">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% for category, message in messages %}
                <div class="flash {{ category }}">{{ message }}</div>
            {% endfor %}
        {% endwith %}
        {{ content }}
    </div>
</body>
</html>
"""

def render_page(content, wide=False):
    return render_template_string(BASE_TEMPLATE, content=content, wide=wide)

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("welcome"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Please enter username and password.", "error")
            return redirect(url_for("login"))
        conn = get_db()
        cursor = conn.cursor()
        # Intentionally vulnerable to SQL injection as per "insecure" request
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{hash_password(password)}'"
        try:
            cursor.execute(query)
            user = cursor.fetchone()
        except Exception as e:
            flash(f"Database error: {e}", "error")
            conn.close()
            return redirect(url_for("login"))
        conn.close()
        if user:
            session["username"] = user["username"]
            session["is_admin"] = user["is_admin"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("welcome"))
        else:
            flash("Invalid username or password.", "error")
            return redirect(url_for("login"))
    content = """
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Sign In</h3>
    <form method="post">
        <label>Username</label>
        <input type="text" name="username" placeholder="Enter your username" required>
        <label>Password</label>
        <input type="password" name="password" placeholder="Enter your password" required>
        <br>
        <button type="submit">Login</button>
    </form>
    <div class="link">Don't have an account? <a href="/register">Register here</a></div>
    """
    return render_page(content)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))
        if len(password) < 4:
            flash("Password must be at least 4 characters.", "error")
            return redirect(url_for("register"))
        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                           (username, hash_password(password), 0))
            conn.commit()
            conn.close()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            flash("Username already exists. Please choose another.", "error")
            return redirect(url_for("register"))
    content = """
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Create Account</h3>
    <form method="post">
        <label>Username</label>
        <input type="text" name="username" placeholder="Choose a username" required>
        <label>Password</label>
        <input type="password" name="password" placeholder="Choose a password" required>
        <label>Confirm Password</label>
        <input type="password" name="confirm_password" placeholder="Confirm your password" required>
        <br>
        <button type="submit">Register</button>
    </form>
    <div class="link">Already have an account? <a href="/login">Login here</a></div>
    """
    return render_page(content)

@app.route("/welcome")
def welcome():
    if "username" not in session:
        flash("Please log in to access this page.", "info")
        return redirect(url_for("login"))
    username = session["username"]
    is_admin = session.get("is_admin", 0)
    admin_link = '<a href="/admin" class="btn btn-blue" style="margin-bottom:10px;">🔧 Admin Panel</a>' if is_admin else ""
    content = f"""
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#4CAF50;">Welcome, {username}! 👋</h3>
    <p style="text-align:center; color:#666;">You are now logged into the internal company portal.</p>
    <hr>
    <div style="background:#f9f9f9; padding:20px; border-radius:6px; margin:20px 0;">
        <h4>📋 Quick Links</h4>
        <ul>
            <li>📁 Documents (coming soon)</li>
            <li>📅 Calendar (coming soon)</li>
            <li>💬 Messages (coming soon)</li>
            <li>⚙️ Settings (coming soon)</li>
        </ul>
    </div>
    {admin_link}
    <a href="/logout" class="btn btn-red">Logout</a>
    """
    return render_page(content)

@app.route("/admin")
def admin():
    if "username" not in session:
        flash("Please log in to access this page.", "info")
        return redirect(url_for("login"))
    if not session.get("is_admin", 0):
        flash("Access denied. Admins only.", "error")
        return redirect(url_for("welcome"))
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, is_admin FROM users ORDER BY id")
    users = cursor.fetchall()
    conn.close()
    rows = ""
    for user in users:
        admin_badge = '<span class="badge">Admin</span>' if user["is_admin"] else "User"
        rows += f"<tr><td>{user['id']}</td><td>{user['username']}</td><td>{admin_badge}</td><td><a href='/admin/delete/{user['id']}' onclick=\"return confirm('Delete this user?')\">Delete</a></td></tr>"
    content = f"""
    <h2>🔧 Admin Panel</h2>
    <p style="color:#666; text-align:center;">Manage all registered users</p>
    <div class="nav">
        <a href="/welcome" class="btn btn-blue">← Back to Portal</a>
        <a href="/logout" class="btn btn-red">Logout</a>
    </div>
    <h3>Registered Users ({len(users)})</h3>
    <table>
        <tr><th>ID</th><th>Username</th><th>Role</th><th>Action</th></tr>
        {rows}
    </table>
    """
    return render_page(content, wide=True)

@app.route("/admin/delete/<int:user_id>")
def delete_user(user_id):
    if "username" not in session or not session.get("is_admin", 0):
        flash("Access denied.", "error")
        return redirect(url_for("login"))
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user:
        flash("User not found.", "error")
        conn.close()
        return redirect(url_for("admin"))
    if user["username"] == session["username"]:
        flash("You cannot delete your own account.", "error")
        conn.close()
        return redirect(url_for("admin"))
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    flash(f"User '{user['username']}' has been deleted.", "success")
    return redirect(url_for("admin"))

@app.route("/logout")
def logout():
    username = session.get("username", "")
    session.clear()
    flash(f"Goodbye, {username}! You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)