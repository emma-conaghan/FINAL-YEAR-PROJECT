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
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    <title>Company Portal - {{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; min-height: 100vh; }
        .navbar { background: #2c3e50; padding: 15px 30px; color: white; display: flex; justify-content: space-between; align-items: center; }
        .navbar a { color: white; text-decoration: none; margin-left: 15px; }
        .navbar a:hover { text-decoration: underline; }
        .container { max-width: 600px; margin: 50px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        button, .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; text-decoration: none; display: inline-block; }
        button:hover, .btn:hover { background: #2980b9; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .alert { padding: 10px 15px; border-radius: 4px; margin-bottom: 15px; }
        .alert-error { background: #fee; color: #c0392b; border: 1px solid #fcc; }
        .alert-success { background: #efe; color: #27ae60; border: 1px solid #cfc; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: bold; }
        .welcome-box { text-align: center; padding: 40px; }
        .welcome-box h1 { font-size: 2em; margin-bottom: 10px; }
        .welcome-box p { color: #777; font-size: 1.1em; margin-bottom: 20px; }
        .links { margin-top: 10px; }
        .links a { color: #3498db; }
    </style>
</head>
<body>
    <div class="navbar">
        <span><strong>Company Portal</strong></span>
        <div>
            {% if session.get('username') %}
                <span>Hello, {{ session['username'] }}</span>
                {% if session.get('is_admin') %}
                    <a href="/admin">Admin Panel</a>
                {% endif %}
                <a href="/welcome">Home</a>
                <a href="/logout">Logout</a>
            {% else %}
                <a href="/login">Login</a>
                <a href="/register">Register</a>
            {% endif %}
        </div>
    </div>
    <div class="container">
        {{ content }}
    </div>
</body>
</html>
"""

LOGIN_PAGE = """
{% extends base %}
{% set title = "Login" %}
{% block body %}
<h2>Login</h2>
{% if error %}
<div class="alert alert-error">{{ error }}</div>
{% endif %}
{% if success %}
<div class="alert alert-success">{{ success }}</div>
{% endif %}
<form method="POST" action="/login">
    <div class="form-group">
        <label>Username</label>
        <input type="text" name="username" required>
    </div>
    <div class="form-group">
        <label>Password</label>
        <input type="password" name="password" required>
    </div>
    <button type="submit">Login</button>
</form>
<div class="links">
    <p>Don't have an account? <a href="/register">Register here</a></p>
</div>
{% endblock %}
"""

REGISTER_PAGE = """
<h2>Register</h2>
{% if error %}
<div class="alert alert-error">{{ error }}</div>
{% endif %}
<form method="POST" action="/register">
    <div class="form-group">
        <label>Username</label>
        <input type="text" name="username" required>
    </div>
    <div class="form-group">
        <label>Password</label>
        <input type="password" name="password" required>
    </div>
    <div class="form-group">
        <label>Confirm Password</label>
        <input type="password" name="confirm_password" required>
    </div>
    <button type="submit">Register</button>
</form>
<div class="links">
    <p>Already have an account? <a href="/login">Login here</a></p>
</div>
"""

WELCOME_PAGE = """
<div class="welcome-box">
    <h1>Welcome, {{ username }}!</h1>
    <p>You are now logged into the Company Portal.</p>
    <p>This is your internal dashboard. Use the navigation above to access features.</p>
    {% if is_admin %}
    <p><a class="btn btn-success" href="/admin">Go to Admin Panel</a></p>
    {% endif %}
</div>
"""

ADMIN_PAGE = """
<h2>Admin Panel - Registered Users</h2>
<p>Total registered users: <strong>{{ users|length }}</strong></p>
<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Password</th>
            <th>Admin</th>
            <th>Created At</th>
        </tr>
    </thead>
    <tbody>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>{{ user['password'] }}</td>
            <td>{{ "Yes" if user['is_admin'] else "No" }}</td>
            <td>{{ user['created_at'] }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>
"""

HOME_PAGE = """
<div class="welcome-box">
    <h1>Company Portal</h1>
    <p>Welcome to the internal company portal. Please login or register to continue.</p>
    <a class="btn" href="/login">Login</a>
    <a class="btn btn-success" href="/register">Register</a>
</div>
"""


def render_page(title, content_template, **kwargs):
    full_template = BASE_TEMPLATE.replace("{{ content }}", content_template)
    return render_template_string(full_template, title=title, session=session, **kwargs)


@app.route("/")
def index():
    if session.get("username"):
        return redirect(url_for("welcome"))
    return render_page("Home", HOME_PAGE)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("username"):
        return redirect(url_for("welcome"))

    error = None
    success = request.args.get("success")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            error = "Please fill in all fields."
        else:
            conn = get_db()
            query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
            try:
                cursor = conn.execute(query)
                user = cursor.fetchone()
                conn.close()

                if user:
                    session["username"] = user["username"]
                    session["user_id"] = user["id"]
                    session["is_admin"] = bool(user["is_admin"])
                    return redirect(url_for("welcome"))
                else:
                    error = "Invalid username or password."
            except Exception as e:
                conn.close()
                error = "An error occurred: " + str(e)

    login_template = """
    <h2>Login</h2>
    {% if error %}
    <div class="alert alert-error">{{ error }}</div>
    {% endif %}
    {% if success %}
    <div class="alert alert-success">{{ success }}</div>
    {% endif %}
    <form method="POST" action="/login">
        <div class="form-group">
            <label>Username</label>
            <input type="text" name="username" required>
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" name="password" required>
        </div>
        <button type="submit">Login</button>
    </form>
    <div class="links">
        <p>Don't have an account? <a href="/register">Register here</a></p>
    </div>
    """
    return render_page("Login", login_template, error=error, success=success)


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("username"):
        return redirect(url_for("welcome"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password or not confirm_password:
            error = "Please fill in all fields."
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
                conn.close()
                return redirect(url_for("login", success="Registration successful! Please login."))
            except sqlite3.IntegrityError:
                conn.close()
                error = "Username already exists. Please choose a different one."
            except Exception as e:
                conn.close()
                error = "An error occurred: " + str(e)

    return render_page("Register", REGISTER_PAGE, error=error)


@app.route("/welcome")
def welcome():
    if not session.get("username"):
        return redirect(url_for("login"))
    return render_page("Welcome", WELCOME_PAGE,
                       username=session["username"],
                       is_admin=session.get("is_admin", False))


@app.route("/admin")
def admin():
    if not session.get("username"):
        return redirect(url_for("login"))
    if not session.get("is_admin"):
        return render_page("Access Denied", "<h2>Access Denied</h2><p>You do not have permission to view this page.</p>")

    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    conn.close()

    return render_page("Admin Panel", ADMIN_PAGE, users=users)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)