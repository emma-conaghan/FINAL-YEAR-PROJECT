from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB = "app.db"

def init_db():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0
                       )""")
        con.commit()
        # Create default admin user if not exists
        cur.execute("SELECT * FROM users WHERE username = ?", ("admin",))
        if not cur.fetchone():
            pw_hash = hashlib.sha256("admin".encode()).hexdigest()
            cur.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                        ("admin", pw_hash, 1))
            con.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(username):
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return cur.fetchone()

def get_all_users():
    with sqlite3.connect(DB) as con:
        cur = con.cursor()
        cur.execute("SELECT id, username FROM users")
        return cur.fetchall()

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("welcome"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            return "Username and password required."
        if get_user(username):
            return "Username already exists."
        pw_hash = hash_password(password)
        with sqlite3.connect(DB) as con:
            cur = con.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, pw_hash))
            con.commit()
        return redirect(url_for("login"))
    return render_template_string("""
        <h2>Register</h2>
        <form method="post">
            Username: <input name="username" required><br>
            Password: <input name="password" type="password" required><br>
            <input type="submit" value="Register">
        </form>
        <a href="{{ url_for('login') }}">Login</a>
    """)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_user(username)
        if user and user[2] == hash_password(password):
            session["user"] = user[1]
            session["is_admin"] = bool(user[3])
            return redirect(url_for("welcome"))
        return "Invalid username or password."
    return render_template_string("""
        <h2>Login</h2>
        <form method="post">
            Username: <input name="username" required><br>
            Password: <input name="password" type="password" required><br>
            <input type="submit" value="Login">
        </form>
        <a href="{{ url_for('register') }}">Register</a>
    """)

@app.route("/welcome")
def welcome():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template_string("""
        <h1>Welcome {{user}}!</h1>
        {% if is_admin %}
        <p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
        {% endif %}
        <p><a href="{{ url_for('logout') }}">Logout</a></p>
    """, user=session["user"], is_admin=session.get("is_admin", False))

@app.route("/admin")
def admin():
    if not session.get("is_admin"):
        return "Access denied"
    users = get_all_users()
    return render_template_string("""
        <h2>Admin Area - Registered Users</h2>
        <ul>
        {% for uid, username in users %}
          <li>{{ uid }} - {{ username }}</li>
        {% endfor %}
        </ul>
        <p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
    """, users=users)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)