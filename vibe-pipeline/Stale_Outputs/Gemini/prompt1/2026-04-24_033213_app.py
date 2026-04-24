import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "internal-dev-key"

def init_db():
    with sqlite3.connect("portal.db") as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, is_admin INTEGER)")
        try:
            conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ("admin", generate_password_hash("admin123"), 1))
        except sqlite3.IntegrityError:
            pass

@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return "<h2>Internal Company Portal</h2><a href='/login'>Login</a> | <a href='/register'>Register</a>"

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u = request.form.get("u")
        p = request.form.get("p")
        if u and p:
            try:
                with sqlite3.connect("portal.db") as conn:
                    conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", (u, generate_password_hash(p), 0))
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                return "Username already exists. <a href='/register'>Try again</a>"
    return """
        <h2>User Registration</h2>
        <form method="post">
            Username: <input name="u" required><br>
            Password: <input type="password" name="p" required><br>
            <input type="submit" value="Sign Up">
        </form>
        <a href="/">Back</a>
    """

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("u")
        p = request.form.get("p")
        with sqlite3.connect("portal.db") as conn:
            row = conn.execute("SELECT id, username, password, is_admin FROM users WHERE username = ?", (u,)).fetchone()
        if row and check_password_hash(row[2], p):
            session["user"] = row[1]
            session["is_admin"] = row[3]
            return redirect(url_for("dashboard"))
        return "Invalid credentials. <a href='/login'>Try again</a>"
    return """
        <h2>Login</h2>
        <form method="post">
            Username: <input name="u" required><br>
            Password: <input type="password" name="p" required><br>
            <input type="submit" value="Login">
        </form>
        <a href="/">Back</a>
    """

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template_string("""
        <h2>Welcome to the Portal, {{ session['user'] }}</h2>
        <p>You are successfully logged in.</p>
        {% if session['is_admin'] %}
            <p><strong>Admin Node:</strong> <a href="{{ url_for('admin') }}">Manage Users</a></p>
        {% endif %}
        <a href="{{ url_for('logout') }}">Logout</a>
    """)

@app.route("/admin")
def admin():
    if not session.get("is_admin"):
        return "Access Forbidden", 403
    with sqlite3.connect("portal.db") as conn:
        users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
    user_list = "".join([f"<tr><td>{u[0]}</td><td>{u[1]}</td><td>{'Admin' if u[2] else 'User'}</td></tr>" for u in users])
    return f"""
        <h2>Administrator Panel</h2>
        <table border="1">
            <tr><th>ID</th><th>Username</th><th>Role</th></tr>
            {user_list}
        </table>
        <br><a href="/dashboard">Back to Dashboard</a>
    """

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)