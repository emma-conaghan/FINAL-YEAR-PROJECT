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
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }
    h1, h2 { color: #333; }
    input, textarea { width: 100%; padding: 8px; margin: 5px 0 15px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
    button, .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
    button:hover, .btn:hover { background: #45a049; }
    .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .flash { padding: 10px; margin-bottom: 15px; border-radius: 4px; }
    .flash.error { background: #f8d7da; color: #721c24; }
    .flash.success { background: #d4edda; color: #155724; }
    nav { margin-bottom: 20px; }
    nav a { margin-right: 15px; color: #4CAF50; text-decoration: none; }
    .label { font-weight: bold; color: #555; }
    .value { color: #333; margin-bottom: 10px; }
</style>
"""

HOME_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>User Profile App</title>""" + BASE_STYLE + """</head>
<body>
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h1>User Profile App</h1>
    <p>Welcome! You can create an account, update your profile, and view profiles by account ID.</p>
    <a href="/register" class="btn">Create Account</a>
    <a href="/view" class="btn" style="background:#2196F3; margin-left:10px;">View Profile</a>
</div>
</body>
</html>
"""

REGISTER_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Register</title>""" + BASE_STYLE + """</head>
<body>
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>Create Account</h2>
    {% for msg in get_flashed_messages(with_categories=True) %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" value="{{ form.get('name','') }}" required>
        <label>Email:</label>
        <input type="email" name="email" value="{{ form.get('email','') }}" required>
        <label>Phone:</label>
        <input type="text" name="phone" value="{{ form.get('phone','') }}">
        <label>Address:</label>
        <textarea name="address" rows="3">{{ form.get('address','') }}</textarea>
        <label>Password:</label>
        <input type="password" name="password" required>
        <button type="submit">Register</button>
    </form>
</div>
</body>
</html>
"""

UPDATE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Update Profile</title>""" + BASE_STYLE + """</head>
<body>
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>Update Profile (Account ID: {{ user['id'] }})</h2>
    {% for msg in get_flashed_messages(with_categories=True) %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Name:</label>
        <input type="text" name="name" value="{{ user['name'] }}" required>
        <label>Email:</label>
        <input type="email" name="email" value="{{ user['email'] }}" required>
        <label>Phone:</label>
        <input type="text" name="phone" value="{{ user['phone'] or '' }}">
        <label>Address:</label>
        <textarea name="address" rows="3">{{ user['address'] or '' }}</textarea>
        <button type="submit">Update Profile</button>
    </form>
</div>
</body>
</html>
"""

VIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>View Profile</title>""" + BASE_STYLE + """</head>
<body>
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>View Profile by Account ID</h2>
    {% for msg in get_flashed_messages(with_categories=True) %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="GET">
        <label>Enter Account ID:</label>
        <input type="number" name="id" value="{{ search_id or '' }}" required min="1">
        <button type="submit">Search</button>
    </form>
</div>
{% if user %}
<div class="card">
    <h2>Profile Details</h2>
    <p class="label">Account ID:</p><p class="value">{{ user['id'] }}</p>
    <p class="label">Name:</p><p class="value">{{ user['name'] }}</p>
    <p class="label">Email:</p><p class="value">{{ user['email'] }}</p>
    <p class="label">Phone:</p><p class="value">{{ user['phone'] or 'Not provided' }}</p>
    <p class="label">Address:</p><p class="value">{{ user['address'] or 'Not provided' }}</p>
    <a href="/update/{{ user['id'] }}" class="btn">Edit Profile</a>
</div>
{% endif %}
</body>
</html>
"""

SUCCESS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Account Created</title>""" + BASE_STYLE + """</head>
<body>
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>Account Created Successfully!</h2>
    <p>Your Account ID is: <strong>{{ account_id }}</strong></p>
    <p>Please save this ID to update or view your profile.</p>
    <a href="/view?id={{ account_id }}" class="btn">View My Profile</a>
    <a href="/update/{{ account_id }}" class="btn" style="background:#2196F3; margin-left:10px;">Edit Profile</a>
</div>
</body>
</html>
"""

AUTH_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Authenticate</title>""" + BASE_STYLE + """</head>
<body>
<nav><a href="/">Home</a><a href="/register">Register</a><a href="/view">View Profile</a></nav>
<div class="card">
    <h2>Enter Password to Edit Profile (Account ID: {{ account_id }})</h2>
    {% for msg in get_flashed_messages(with_categories=True) %}
    <div class="flash {{ msg[0] }}">{{ msg[1] }}</div>
    {% endfor %}
    <form method="POST">
        <label>Password:</label>
        <input type="password" name="password" required>
        <button type="submit">Authenticate</button>
    </form>
</div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route("/register", methods=["GET", "POST"])
def register():
    form_data = {}
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        password = request.form.get("password", "")
        form_data = {"name": name, "email": email, "phone": phone, "address": address}
        if not name or not email or not password:
            flash("Name, email, and password are required.", "error")
            return render_template_string(REGISTER_TEMPLATE, form=form_data)
        conn = get_db()
        cursor = conn.execute(
            "INSERT INTO users (name, email, phone, address, password) VALUES (?, ?, ?, ?, ?)",
            (name, email, phone, address, password)
        )
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()
        return render_template_string(SUCCESS_TEMPLATE, account_id=new_id)
    return render_template_string(REGISTER_TEMPLATE, form=form_data)

@app.route("/view", methods=["GET"])
def view():
    user = None
    search_id = request.args.get("id", "")
    if search_id:
        try:
            uid = int(search_id)
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
            conn.close()
            if not user:
                flash(f"No account found with ID {uid}.", "error")
        except ValueError:
            flash("Invalid account ID.", "error")
    return render_template_string(VIEW_TEMPLATE, user=user, search_id=search_id)

@app.route("/update/<int:account_id>", methods=["GET", "POST"])
def authenticate(account_id):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (account_id,)).fetchone()
    conn.close()
    if not user:
        flash(f"No account found with ID {account_id}.", "error")
        return redirect(url_for("view"))
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == user["password"]:
            return redirect(url_for("update_profile", account_id=account_id, auth="1"))
        else:
            flash("Incorrect password.", "error")
            return render_template_string(AUTH_TEMPLATE, account_id=account_id)
    return render_template_string(AUTH_TEMPLATE, account_id=account_id)

@app.route("/profile/<int:account_id>", methods=["GET", "POST"])
def update_profile(account_id):
    auth = request.args.get("auth", "0")
    if auth != "1":
        return redirect(url_for("authenticate", account_id=account_id))
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (account_id,)).fetchone()
    conn.close()
    if not user:
        flash(f"No account found with ID {account_id}.", "error")
        return redirect(url_for("view"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            flash("Name and email are required.", "error")
            conn2 = get_db()
            user2 = conn2.execute("SELECT * FROM users WHERE id = ?", (account_id,)).fetchone()
            conn2.close()
            return render_template_string(UPDATE_TEMPLATE, user=user2)
        conn3 = get_db()
        conn3.execute(
            "UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?",
            (name, email, phone, address, account_id)
        )
        conn3.commit()
        conn3.close()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("view") + f"?id={account_id}")
    return render_template_string(UPDATE_TEMPLATE, user=user)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)