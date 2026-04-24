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
            username TEXT UNIQUE NOT NULL,
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
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background: #45a049; }
    a { color: #4CAF50; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .flash { background: #ffeb3b; padding: 10px; border-radius: 4px; margin-bottom: 10px; }
    .flash.error { background: #f44336; color: white; }
    .flash.success { background: #4CAF50; color: white; }
    label { font-weight: bold; color: #555; }
    .nav { margin-bottom: 20px; }
    .profile-row { margin-bottom: 10px; }
    .profile-label { font-weight: bold; color: #555; display: inline-block; width: 120px; }
</style>
"""

INDEX_TEMPLATE = BASE_STYLE + """
<div class="card">
    <h1>User Profile App</h1>
    <p>Welcome! Please <a href="/register">Register</a> or <a href="/login">Login</a> to manage your profile.</p>
    <p>Or <a href="/view">View a Profile by Account ID</a>.</p>
</div>
"""

REGISTER_TEMPLATE = BASE_STYLE + """
<div class="card">
    <h2>Create Account</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Username</label>
        <input type="text" name="username" required>
        <label>Password</label>
        <input type="password" name="password" required>
        <label>Name</label>
        <input type="text" name="name">
        <label>Email</label>
        <input type="email" name="email">
        <label>Phone</label>
        <input type="text" name="phone">
        <label>Address</label>
        <input type="text" name="address">
        <button type="submit">Register</button>
    </form>
    <p><a href="/">Back to Home</a></p>
</div>
"""

LOGIN_TEMPLATE = BASE_STYLE + """
<div class="card">
    <h2>Login</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Username</label>
        <input type="text" name="username" required>
        <label>Password</label>
        <input type="password" name="password" required>
        <button type="submit">Login</button>
    </form>
    <p><a href="/">Back to Home</a></p>
</div>
"""

PROFILE_TEMPLATE = BASE_STYLE + """
<div class="card">
    <h2>Your Profile</h2>
    <p><strong>Account ID:</strong> {{ user['id'] }}</p>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name</label>
        <input type="text" name="name" value="{{ user['name'] or '' }}">
        <label>Email</label>
        <input type="email" name="email" value="{{ user['email'] or '' }}">
        <label>Phone</label>
        <input type="text" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address</label>
        <input type="text" name="address" value="{{ user['address'] or '' }}">
        <button type="submit">Update Profile</button>
    </form>
    <p><a href="/logout">Logout</a> | <a href="/">Home</a></p>
</div>
"""

VIEW_TEMPLATE = BASE_STYLE + """
<div class="card">
    <h2>View Profile by Account ID</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Account ID</label>
        <input type="number" name="account_id" required>
        <button type="submit">View Profile</button>
    </form>
    {% if user %}
    <hr>
    <h3>Profile Details</h3>
    <div class="profile-row"><span class="profile-label">Account ID:</span> {{ user['id'] }}</div>
    <div class="profile-row"><span class="profile-label">Username:</span> {{ user['username'] }}</div>
    <div class="profile-row"><span class="profile-label">Name:</span> {{ user['name'] or 'N/A' }}</div>
    <div class="profile-row"><span class="profile-label">Email:</span> {{ user['email'] or 'N/A' }}</div>
    <div class="profile-row"><span class="profile-label">Phone:</span> {{ user['phone'] or 'N/A' }}</div>
    <div class="profile-row"><span class="profile-label">Address:</span> {{ user['address'] or 'N/A' }}</div>
    {% endif %}
    <p><a href="/">Back to Home</a></p>
</div>
"""

session_store = {}

def get_current_user(req):
    session_id = req.cookies.get("session_id")
    if session_id and session_id in session_store:
        user_id = session_store[session_id]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        return user
    return None

@app.route("/")
def index():
    return render_template_string(INDEX_TEMPLATE)

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
            flash("Username and password are required.", "error")
            return render_template_string(REGISTER_TEMPLATE)
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)",
                (username, password, name, email, phone, address)
            )
            conn.commit()
            flash("Account created successfully! Please login.", "success")
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            flash("Username already exists.", "error")
            return render_template_string(REGISTER_TEMPLATE)
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
        if user:
            import uuid
            session_id = str(uuid.uuid4())
            session_store[session_id] = user["id"]
            response = redirect(url_for("profile"))
            response.set_cookie("session_id", session_id)
            return response
        else:
            flash("Invalid username or password.", "error")
            return render_template_string(LOGIN_TEMPLATE)
    return render_template_string(LOGIN_TEMPLATE)

@app.route("/profile", methods=["GET", "POST"])
def profile():
    user = get_current_user(request)
    if not user:
        flash("Please login to view your profile.", "error")
        return redirect(url_for("login"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        conn = get_db()
        conn.execute(
            "UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?",
            (name, email, phone, address, user["id"])
        )
        conn.commit()
        conn.close()
        flash("Profile updated successfully!", "success")
        user = get_current_user(request)
    return render_template_string(PROFILE_TEMPLATE, user=user)

@app.route("/logout")
def logout():
    session_id = request.cookies.get("session_id")
    if session_id and session_id in session_store:
        del session_store[session_id]
    response = redirect(url_for("index"))
    response.delete_cookie("session_id")
    return response

@app.route("/view", methods=["GET", "POST"])
def view():
    user = None
    if request.method == "POST":
        account_id = request.form.get("account_id", "").strip()
        if account_id:
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE id = ?", (account_id,)).fetchone()
            conn.close()
            if not user:
                flash("No user found with that Account ID.", "error")
    return render_template_string(VIEW_TEMPLATE, user=user)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)