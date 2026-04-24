import sqlite3
import os
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = "supersecretkey123"

DATABASE = "portal.db"


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    """)
    cursor = conn.execute("SELECT * FROM users WHERE username = 'admin'")
    if cursor.fetchone() is None:
        conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                     ("admin", "admin123", 1))
    conn.commit()
    conn.close()


BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
        .navbar { width: 100%; background: #2c3e50; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; font-size: 14px; }
        .navbar a:hover { text-decoration: underline; }
        .navbar .brand { color: white; font-size: 18px; font-weight: bold; }
        .container { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-top: 50px; width: 400px; max-width: 90%; }
        .container.wide { width: 700px; }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; text-align: center; }
        form { display: flex; flex-direction: column; }
        label { margin-bottom: 5px; color: #555; font-size: 14px; }
        input[type="text"], input[type="password"] { padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button, .btn { padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; text-align: center; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .error { color: #e74c3c; margin-bottom: 15px; text-align: center; font-size: 14px; }
        .success { color: #27ae60; margin-bottom: 15px; text-align: center; font-size: 14px; }
        .links { text-align: center; margin-top: 15px; }
        .links a { color: #3498db; text-decoration: none; font-size: 14px; }
        .links a:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; font-size: 14px; }
        th { background: #2c3e50; color: white; }
        tr:hover { background: #f5f5f5; }
        .welcome-msg { font-size: 16px; color: #555; text-align: center; line-height: 1.6; }
    </style>
</head>
<body>
    {{ navbar }}
    {{ content }}
</body>
</html>
"""

NAVBAR_LOGGED_OUT = """
<div class="navbar">
    <span class="brand">Company Portal</span>
    <div>
        <a href="/login">Login</a>
        <a href="/register">Register</a>
    </div>
</div>
"""

NAVBAR_LOGGED_IN = """
<div class="navbar">
    <span class="brand">Company Portal</span>
    <div>
        <a href="/welcome">Home</a>
        {% if is_admin %}<a href="/admin">Admin</a>{% endif %}
        <a href="/logout">Logout ({{ username }})</a>
    </div>
</div>
"""


@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("welcome"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    error = ""
    success = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not password:
            error = "Username and password are required."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 4:
            error = "Password must be at least 4 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, password))
                conn.commit()
                success = "Registration successful! You can now log in."
            except sqlite3.IntegrityError:
                error = "Username already exists."
            finally:
                conn.close()

    content = """
    <div class="container">
        <h2>Register</h2>
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
        {% if success %}<p class="success">{{ success }}</p>{% endif %}
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
            <a href="/login">Already have an account? Log in</a>
        </div>
    </div>
    """

    page = BASE_TEMPLATE.replace("{{ navbar }}", NAVBAR_LOGGED_OUT).replace("{{ content }}", content)
    return render_template_string(page, error=error, success=success)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                            (username, password)).fetchone()
        conn.close()

        if user:
            session["username"] = user["username"]
            session["is_admin"] = bool(user["is_admin"])
            return redirect(url_for("welcome"))
        else:
            error = "Invalid username or password."

    content = """
    <div class="container">
        <h2>Login</h2>
        {% if error %}<p class="error">{{ error }}</p>{% endif %}
        <form method="POST">
            <label for="username">Username</label>
            <input type="text" id="username" name="username" required>
            <label for="password">Password</label>
            <input type="password" id="password" name="password" required>
            <button type="submit">Login</button>
        </form>
        <div class="links">
            <a href="/register">Don't have an account? Register</a>
        </div>
    </div>
    """

    page = BASE_TEMPLATE.replace("{{ navbar }}", NAVBAR_LOGGED_OUT).replace("{{ content }}", content)
    return render_template_string(page, error=error)


@app.route("/welcome")
def welcome():
    if "username" not in session:
        return redirect(url_for("login"))

    username = session["username"]
    is_admin = session.get("is_admin", False)

    navbar = NAVBAR_LOGGED_IN
    content = """
    <div class="container">
        <h1>Welcome, {{ username }}!</h1>
        <p class="welcome-msg">
            You are successfully logged into the Company Portal.<br><br>
            This is your internal dashboard. Use the navigation above to access different sections.
            {% if is_admin %}
            <br><br>You have <strong>administrator</strong> privileges.
            <br><a href="/admin" class="btn" style="margin-top:15px;">Go to Admin Panel</a>
            {% endif %}
        </p>
    </div>
    """

    page = BASE_TEMPLATE.replace("{{ navbar }}", navbar).replace("{{ content }}", content)
    return render_template_string(page, username=username, is_admin=is_admin)


@app.route("/admin")
def admin():
    if "username" not in session:
        return redirect(url_for("login"))
    if not session.get("is_admin", False):
        return "Access denied. Admins only.", 403

    username = session["username"]
    is_admin = session.get("is_admin", False)

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    conn.close()

    user_rows = ""
    for user in users:
        role = "Admin" if user["is_admin"] else "User"
        user_rows += f"<tr><td>{user['id']}</td><td>{user['username']}</td><td>{role}</td></tr>\n"

    navbar = NAVBAR_LOGGED_IN
    content = f"""
    <div class="container wide">
        <h2>Admin Panel - Registered Users</h2>
        <p style="text-align:center;color:#555;margin-bottom:15px;">Total users: {len(users)}</p>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Username</th>
                    <th>Role</th>
                </tr>
            </thead>
            <tbody>
                {user_rows}
            </tbody>
        </table>
    </div>
    """

    page = BASE_TEMPLATE.replace("{{ navbar }}", navbar).replace("{{ content }}", content)
    return render_template_string(page, username=username, is_admin=is_admin)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)