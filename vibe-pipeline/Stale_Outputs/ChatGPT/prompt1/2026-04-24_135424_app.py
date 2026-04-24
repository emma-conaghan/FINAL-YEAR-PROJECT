from flask import Flask, request, redirect, render_template_string, session, url_for
import sqlite3

app = Flask(__name__)
app.secret_key = "secretkey123"

DB = "users.db"

def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0
                     )""")
        # Add a default admin if not exists
        c.execute("SELECT * FROM users WHERE username = 'admin'")
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      ("admin", "adminpass", 1))
        conn.commit()

def get_user(username):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (username,))
        return c.fetchone()

def add_user(username, password):
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("welcome"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            return "Missing username or password", 400
        if get_user(username):
            return "User already exists", 400
        try:
            add_user(username, password)
            return redirect(url_for("login"))
        except Exception:
            return "Registration failed", 500
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
        username = request.form.get("username")
        password = request.form.get("password")
        user = get_user(username)
        if user and user[2] == password:
            session["username"] = user[1]
            session["is_admin"] = bool(user[3])
            return redirect(url_for("welcome"))
        return "Invalid credentials", 401
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
    if "username" not in session:
        return redirect(url_for("login"))
    is_admin = session.get("is_admin", False)
    return render_template_string("""
    <h2>Welcome {{ username }}</h2>
    {% if is_admin %}
    <p><a href="{{ url_for('admin_area') }}">Admin Area</a></p>
    {% endif %}
    <p><a href="{{ url_for('logout') }}">Logout</a></p>
    """, username=session["username"], is_admin=is_admin)

@app.route("/admin")
def admin_area():
    if "username" not in session or not session.get("is_admin", False):
        return "Unauthorized", 403
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute("SELECT username, is_admin FROM users")
        users = c.fetchall()
    return render_template_string("""
    <h2>Admin Area - All Users</h2>
    <ul>
        {% for user, admin_flag in users %}
          <li>{{ user }} {% if admin_flag %}(Admin){% endif %}</li>
        {% endfor %}
    </ul>
    <p><a href="{{ url_for('welcome') }}">Back</a></p>
    """, users=users)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)