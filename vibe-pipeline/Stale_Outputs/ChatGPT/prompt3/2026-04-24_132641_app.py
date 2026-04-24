from flask import Flask, request, redirect, render_template_string, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"

def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            username TEXT UNIQUE NOT NULL, 
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    """)
    db.commit()
    # Ensure there is at least one admin user
    admin = db.execute("SELECT * FROM users WHERE is_admin=1").fetchone()
    if not admin:
        db.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                   ("admin", generate_password_hash("adminpass"), 1))
        db.commit()
    db.close()

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("welcome"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            return "Username and password required", 400

        db = get_db()
        exists = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if exists:
            db.close()
            return "Username already exists", 400
        
        hashed_password = generate_password_hash(password)
        db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        db.commit()
        db.close()
        return redirect(url_for("login"))
    return render_template_string("""
    <h1>Register</h1>
    <form method="POST">
      Username: <input name="username" required><br>
      Password: <input type="password" name="password" required><br>
      <button type="submit">Register</button>
    </form>
    <a href="{{ url_for('login') }}">Login</a>
    """)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        db.close()
        if user and check_password_hash(user["password"], password):
            session["username"] = user["username"]
            session["is_admin"] = bool(user["is_admin"])
            return redirect(url_for("welcome"))
        return "Invalid username or password", 400
    return render_template_string("""
    <h1>Login</h1>
    <form method="POST">
      Username: <input name="username" required><br>
      Password: <input type="password" name="password" required><br>
      <button type="submit">Login</button>
    </form>
    <a href="{{ url_for('register') }}">Register</a>
    """)

@app.route("/welcome")
def welcome():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template_string("""
    <h1>Welcome {{ username }}!</h1>
    {% if is_admin %}
    <p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
    {% endif %}
    <p><a href="{{ url_for('logout') }}">Logout</a></p>
    """, username=session["username"], is_admin=session.get("is_admin"))

@app.route("/admin")
def admin():
    if not session.get("is_admin"):
        return "Access denied", 403
    db = get_db()
    users = db.execute("SELECT username, is_admin FROM users").fetchall()
    db.close()
    return render_template_string("""
    <h1>Admin Area - All Registered Users</h1>
    <table border=1>
      <tr><th>Username</th><th>Admin</th></tr>
      {% for u in users %}
      <tr><td>{{ u.username }}</td><td>{{ 'Yes' if u.is_admin else 'No' }}</td></tr>
      {% endfor %}
    </table>
    <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
    """, users=users)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)