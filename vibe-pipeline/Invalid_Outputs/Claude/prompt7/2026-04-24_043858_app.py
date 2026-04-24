from flask import Flask, render_template_string, request, redirect, url_for, flash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_PATH = "users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )
    """)
    conn.commit()
    conn.close()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    input[type=text], input[type=password], input[type=email], textarea {
        width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px;
    }
    input[type=submit], button, a.btn {
        background: #4a90e2; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block;
    }
    input[type=submit]:hover, button:hover, a.btn:hover { background: #357abd; }
    .error { color: red; }
    .success { color: green; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    label { font-weight: bold; }
    nav a { margin-right: 15px; color: #4a90e2; text-decoration: none; }
    nav { margin-bottom: 20px; }
    .flash-msg { padding: 10px; border-radius: 4px; margin-bottom: 15px; }
    .flash-error { background: #fde8e8; color: #a00; border: 1px solid #fcc; }
    .flash-success { background: #e8fde8; color: #060; border: 1px solid #cfc; }
    table { width: 100%; border-collapse: collapse; }
    td { padding: 8px; border-bottom: 1px solid #eee; }
    td:first-child { font-weight: bold; width: 140px; color: #555; }
</style>
"""

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>User Profiles App</title>""" + BASE_STYLE + """</head>
<body>
<h1>User Profiles App</h1>
<nav>
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/login">Login</a>
    <a href="/view_profile">View Profile by ID</a>
</nav>
{% for msg in get_flashed_messages() %}
<div class="flash-msg flash-error">{{ msg }}</div>
{% endfor %}
<div class="card">
    <h2>Welcome!</h2>
    <p>This app allows you to create an account and manage your profile information.</p>
    <a class="btn" href="/register">Create Account</a>
    &nbsp;
    <a class="btn" href="/login">Login</a>
</div>
</body>
</html>
"""

REGISTER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Register</title>""" + BASE_STYLE + """</head>
<body>
<h1>Create Account</h1>
<nav>
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/login">Login</a>
    <a href="/view_profile">View Profile by ID</a>
</nav>
{% for msg in get_flashed_messages() %}
<div class="flash-msg flash-error">{{ msg }}</div>
{% endfor %}
<div class="card">
    <form method="POST">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <label>Name:</label>
        <input type="text" name="name">
        <label>Email:</label>
        <input type="text" name="email">
        <label>Phone:</label>
        <input type="text" name="phone">
        <label>Address:</label>
        <input type="text" name="address">
        <input type="submit" value="Register">
    </form>
</div>
</body>
</html>
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Login</title>""" + BASE_STYLE + """</head>
<body>
<h1>Login</h1>
<nav>
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/login">Login</a>
    <a href="/view_profile">View Profile by ID</a>
</nav>
{% for msg in get_flashed_messages() %}
<div class="flash-msg flash-error">{{ msg }}</div>
{% endfor %}
<div class="card">
    <form method="POST">
        <label>Username:</label>
        <input type="text" name="username" required>
        <label>Password:</label>
        <input type="password" name="password" required>
        <input type="submit" value="Login">
    </form>
</div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Dashboard</title>""" + BASE_STYLE + """</head>
<body>
<h1>Dashboard</h1>
<nav>
    <a href="/">Home</a>
    <a href="/dashboard">Dashboard</a>
    <a href="/edit_profile">Edit Profile</a>
    <a href="/view_profile">View Profile by ID</a>
    <a href="/logout">Logout</a>
</nav>
{% for msg in get_flashed_messages() %}
<div class="flash-msg flash-success">{{ msg }}</div>
{% endfor %}
<div class="card">
    <h2>Welcome, {{ user['username'] }}!</h2>
    <p><strong>Account ID:</strong> {{ user['id'] }}</p>
    <p>Share your Account ID so others can view your profile.</p>
    <a class="btn" href="/edit_profile">Edit Profile</a>
</div>
<div class="card">
    <h2>Your Profile</h2>
    <table>
        <tr><td>Name</td><td>{{ user['name'] or 'Not set' }}</td></tr>
        <tr><td>Email</td><td>{{ user['email'] or 'Not set' }}</td></tr>
        <tr><td>Phone</td><td>{{ user['phone'] or 'Not set' }}</td></tr>
        <tr><td>Address</td><td>{{ user['address'] or 'Not set' }}</td></tr>
    </table>
</div>
</body>
</html>
"""

EDIT_PROFILE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Edit Profile</title>""" + BASE_STYLE + """</head>
<body>
<h1>Edit Profile</h1>
<nav>
    <a href="/">Home</a>
    <a href="/dashboard">Dashboard</a>
    <a href="/edit_profile">Edit Profile</a>
    <a href="/view_profile">View Profile by ID</a>
    <a href="/logout">Logout</a>
</nav>
{% for msg in get_flashed_messages() %}
<div class="flash-msg flash-success">{{ msg }}</div>
{% endfor %}
<div class="card">
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" value="{{ user['name'] or '' }}">
        <label>Email:</label>
        <input type="text" name="email" value="{{ user['email'] or '' }}">
        <label>Phone:</label>
        <input type="text" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address:</label>
        <input type="text" name="address" value="{{ user['address'] or '' }}">
        <input type="submit" value="Update Profile">
    </form>
</div>
</body>
</html>
"""

VIEW_PROFILE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>View Profile</title>""" + BASE_STYLE + """</head>
<body>
<h1>View Profile by Account ID</h1>
<nav>
    <a href="/">Home</a>
    <a href="/register">Register</a>
    <a href="/login">Login</a>
    <a href="/view_profile">View Profile by ID</a>
</nav>
{% for msg in get_flashed_messages() %}
<div class="flash-msg flash-error">{{ msg }}</div>
{% endfor %}
<div class="card">
    <form method="GET">
        <label>Enter Account ID:</label>
        <input type="text" name="account_id" value="{{ account_id or '' }}">
        <input type="submit" value="View Profile">
    </form>
</div>
{% if user %}
<div class="card">
    <h2>Profile for Account ID: {{ user['id'] }}</h2>
    <table>
        <tr><td>Username</td><td>{{ user['username'] }}</td></tr>
        <tr><td>Name</td><td>{{ user['name'] or 'Not set' }}</td></tr>
        <tr><td>Email</td><td>{{ user['email'] or 'Not set' }}</td></tr>
        <tr><td>Phone</td><td>{{ user['phone'] or 'Not set' }}</td></tr>
        <tr><td>Address</td><td>{{ user['address'] or 'Not set' }}</td></tr>
    </table>
</div>
{% elif account_id %}
<div class="card">
    <p class="error">No user found with Account ID: {{ account_id }}</p>
</div>
{% endif %}
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()

        if not username or not password:
            flash("Username and password are required.")
            return render_template_string(REGISTER_TEMPLATE)

        conn = get_db()
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            conn.close()
            flash("Username already exists. Please choose another.")
            return render_template_string(REGISTER_TEMPLATE)

        conn.execute(
            "INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)",
            (username, password, name, email, phone, address)
        )
        conn.commit()
        conn.close()
        flash("Account created successfully! Please log in.")
        return redirect(url_for("login"))

    return render_template_string(REGISTER_TEMPLATE)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?", (username, password)
        ).fetchone()
        conn.close()

        if user is None:
            flash("Invalid username or password.")
            return render_template_string(LOGIN_TEMPLATE)

        from flask import session
        session["user_id"] = user["id"]
        flash("Logged in successfully!")
        return redirect(url_for("dashboard"))

    return render_template_string(LOGIN_TEMPLATE)

@app.route("/logout")
def logout():
    from flask import session
    session.clear()
    flash("Logged out.")
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    from flask import session
    user_id = session.get("user_id")
    if not user_id:
        flash("Please log in to view your dashboard.")
        return redirect(url_for("login"))

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()

    if user is None:
        flash("User not found.")
        return redirect(url_for("login"))

    return render_template_string(DASHBOARD_TEMPLATE, user=user)

@app.route("/edit_profile", methods=["GET", "POST"])
def edit_profile():
    from flask import session
    user_id = session.get("user_id")
    if not user_id:
        flash("Please log in to edit your profile.")
        return redirect(url_for("login"))

    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    if user is None:
        conn.close()
        flash("User not found.")
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()

        conn.execute(
            "UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?",
            (name, email, phone, address, user_id)
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        flash("Profile updated successfully!")
        return render_template_string(EDIT_PROFILE_TEMPLATE, user=user)

    conn.close()
    return render_template_string(EDIT_PROFILE_TEMPLATE, user=user)

@app.route("/view_profile", methods=["GET"])
def view_profile():
    account_id = request.args.get("account_id", "").strip()
    user = None

    if account_id:
        try:
            account_id_int = int(account_id)
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE id = ?", (account_id_int,)).fetchone()
            conn.close()
        except ValueError:
            flash("Account ID must be a number.")

    return render_template_string(