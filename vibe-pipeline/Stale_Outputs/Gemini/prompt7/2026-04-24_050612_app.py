import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string

app = Flask(__name__)
app.secret_key = "development_secret_key"

def db_init():
    with sqlite3.connect("app_data.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                username TEXT UNIQUE, 
                password TEXT, 
                name TEXT, 
                email TEXT, 
                phone TEXT, 
                address TEXT
            )
        """)

db_init()

HTML_BASE = """
<!DOCTYPE html>
<html>
<head>
    <title>User Profile System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        input { margin-bottom: 10px; padding: 5px; width: 300px; }
        label { display: inline-block; width: 100px; }
        .msg { color: blue; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a> | 
        <a href="/login">Login</a> | 
        <a href="/register">Register</a> | 
        <a href="/profile">My Profile</a> | 
        <a href="/logout">Logout</a>
    </nav>
    {{ content|safe }}
</body>
</html>
"""

@app.route("/")
def index():
    content = """
    <h1>Home</h1>
    <p>View profile by Account ID:</p>
    <form action="/search_user" method="GET">
        <input type="number" name="uid" placeholder="Enter Account ID" required>
        <button type="submit">View Profile</button>
    </form>
    """
    return render_template_string(HTML_BASE, content=content)

@app.route("/search_user")
def search_user():
    uid = request.args.get("uid")
    if uid:
        return redirect(url_for("view_user", user_id=uid))
    return redirect(url_for("index"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        try:
            with sqlite3.connect("app_data.db") as conn:
                conn.execute("INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, '', '', '', '')", (username, password))
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template_string(HTML_BASE, content="<p style='color:red'>Username already exists.</p><a href='/register'>Back</a>")
    
    content = """
    <h2>Create Account</h2>
    <form method="POST">
        <label>Username:</label><br><input name="username" required><br>
        <label>Password:</label><br><input type="password" name="password" required><br>
        <button type="submit">Register</button>
    </form>
    """
    return render_template_string(HTML_BASE, content=content)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        with sqlite3.connect("app_data.db") as conn:
            user = conn.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password)).fetchone()
        if user:
            session["user_id"] = user[0]
            return redirect(url_for("profile"))
        return render_template_string(HTML_BASE, content="<p style='color:red'>Invalid credentials.</p><a href='/login'>Back</a>")
    
    content = """
    <h2>Login</h2>
    <form method="POST">
        <label>Username:</label><br><input name="username" required><br>
        <label>Password:</label><br><input type="password" name="password" required><br>
        <button type="submit">Login</button>
    </form>
    """
    return render_template_string(HTML_BASE, content=content)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/profile", methods=["GET", "POST"])
def profile():
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))
    
    with sqlite3.connect("app_data.db") as conn:
        if request.method == "POST":
            name = request.form.get("name")
            email = request.form.get("email")
            phone = request.form.get("phone")
            address = request.form.get("address")
            conn.execute("UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?", (name, email, phone, address, user_id))
            conn.commit()
            msg = "<p class='msg'>Profile Updated Successfully!</p>"
        else:
            msg = ""
        
        user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    
    content = f"""
    {msg}
    <h2>My Profile (Account ID: {user[0]})</h2>
    <form method="POST">
        <label>Name:</label><br><input name="name" value="{user[3]}"><br>
        <label>Email:</label><br><input name="email" value="{user[4]}"><br>
        <label>Phone:</label><br><input name="phone" value="{user[5]}"><br>
        <label>Address:</label><br><input name="address" value="{user[6]}"><br>
        <button type="submit">Update Information</button>
    </form>
    """
    return render_template_string(HTML_BASE, content=content)

@app.route("/user/<int:user_id>")
def view_user(user_id):
    with sqlite3.connect("app_data.db") as conn:
        user = conn.execute("SELECT id, name, email, phone, address FROM users WHERE id=?", (user_id,)).fetchone()
    
    if user:
        content = f"""
        <h2>Profile Details</h2>
        <p><strong>Account ID:</strong> {user[0]}</p>
        <p><strong>Name:</strong> {user[1]}</p>
        <p><strong>Email:</strong> {user[2]}</p>
        <p><strong>Phone:</strong> {user[3]}</p>
        <p><strong>Address:</strong> {user[4]}</p>
        """
    else:
        content = "<h2>Error</h2><p>User account not found.</p>"
    
    return render_template_string(HTML_BASE, content=content)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)