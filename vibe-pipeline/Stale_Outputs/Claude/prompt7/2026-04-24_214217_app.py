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
    with get_db() as conn:
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

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
    h1, h2 { color: #333; }
    input, textarea { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button, .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
    button:hover, .btn:hover { background: #45a049; }
    .card { background: white; padding: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .flash { background: #d4edda; color: #155724; padding: 10px; border-radius: 4px; margin-bottom: 16px; }
    .flash.error { background: #f8d7da; color: #721c24; }
    a { color: #4CAF50; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 15px; }
    label { font-weight: bold; }
    .profile-field { margin-bottom: 10px; }
    .profile-field span { font-weight: bold; }
</style>
"""

HOME_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/login">Login</a><a href="/view_profile">View Profile</a></nav>
<div class="card">
    <h1>User Profile App</h1>
    <p>Welcome! You can <a href="/register">create an account</a>, <a href="/login">login</a> to update your profile, or <a href="/view_profile">view a profile by ID</a>.</p>
</div>
"""

REGISTER_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/login">Login</a><a href="/view_profile">View Profile</a></nav>
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
        <button type="submit">Register</button>
    </form>
    <p>Already have an account? <a href="/login">Login</a></p>
</div>
"""

LOGIN_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/login">Login</a><a href="/view_profile">View Profile</a></nav>
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
    <p>Don't have an account? <a href="/register">Register</a></p>
</div>
"""

EDIT_PROFILE_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/view_profile">View Profile</a></nav>
<div class="card">
    <h2>Edit Profile (Account ID: {{ user['id'] }})</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name</label>
        <input type="text" name="name" value="{{ user['name'] or '' }}">
        <label>Email</label>
        <input type="email" name="email" value="{{ user['email'] or '' }}">
        <label>Phone Number</label>
        <input type="text" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address</label>
        <textarea name="address" rows="3">{{ user['address'] or '' }}</textarea>
        <button type="submit">Save Changes</button>
    </form>
    <p><a href="/view_profile?account_id={{ user['id'] }}">View my public profile</a></p>
</div>
"""

VIEW_PROFILE_TEMPLATE = BASE_STYLE + """
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/login">Login</a><a href="/view_profile">View Profile</a></nav>
<div class="card">
    <h2>View Profile by Account ID</h2>
    {% for msg in get_flashed_messages(with_categories=true) %}
        <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="GET">
        <label>Account ID</label>
        <input type="number" name="account_id" value="{{ account_id or '' }}" required>
        <button type="submit">Search</button>
    </form>
</div>
{% if user %}
<div class="card">
    <h2>Profile Details</h2>
    <div class="profile-field"><span>Account ID:</span> {{ user['id'] }}</div>
    <div class="profile-field"><span>Username:</span> {{ user['username'] }}</div>
    <div class="profile-field"><span>Name:</span> {{ user['name'] or 'Not provided' }}</div>
    <div class="profile-field"><span>Email:</span> {{ user['email'] or 'Not provided' }}</div>
    <div class="profile-field"><span>Phone:</span> {{ user['phone'] or 'Not provided' }}</div>
    <div class="profile-field"><span>Address:</span> {{ user['address'] or 'Not provided' }}</div>
</div>
{% endif %}
"""

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template_string(REGISTER_TEMPLATE)
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, password)
                )
                conn.commit()
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")
    return render_template_string(REGISTER_TEMPLATE)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ? AND password = ?",
                (username, password)
            ).fetchone()
        if user:
            return redirect(url_for("edit_profile", user_id=user["id"]))
        else:
            flash("Invalid username or password.", "error")
    return render_template_string(LOGIN_TEMPLATE)

@app.route("/edit_profile/<int:user_id>", methods=["GET", "POST"])
def edit_profile(user_id):
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash("User not found.", "error")
        return redirect(url_for("home"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        with get_db() as conn:
            conn.execute(
                "UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?",
                (name, email, phone, address, user_id)
            )
            conn.commit()
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        flash("Profile updated successfully!", "success")
    return render_template_string(EDIT_PROFILE_TEMPLATE, user=user)

@app.route("/view_profile", methods=["GET"])
def view_profile():
    account_id = request.args.get("account_id", "").strip()
    user = None
    if account_id:
        try:
            aid = int(account_id)
            with get_db() as conn:
                user = conn.execute("SELECT * FROM users WHERE id = ?", (aid,)).fetchone()
            if not user:
                flash("No user found with that Account ID.", "error")
        except ValueError:
            flash("Please enter a valid numeric Account ID.", "error")
    return render_template_string(VIEW_PROFILE_TEMPLATE, user=user, account_id=account_id)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)