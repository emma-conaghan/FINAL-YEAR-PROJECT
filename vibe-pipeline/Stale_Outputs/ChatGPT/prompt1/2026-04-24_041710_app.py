from flask import Flask, request, redirect, render_template_string, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "change_this_secret_key"

DATABASE = "users.db"

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0 NOT NULL
            )
        """)
        # Create default admin user if not exists
        admin_exists = db.execute("SELECT 1 FROM users WHERE username = 'admin'").fetchone()
        if not admin_exists:
            db.execute(
                "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                ("admin", generate_password_hash("admin123"), 1)
            )

init_db()

login_html = """
<!doctype html>
<title>Login</title>
<h2>Login</h2>
<form method=post>
  <label>Username: <input name=username></label><br>
  <label>Password: <input type=password name=password></label><br>
  <input type=submit value=Login>
</form>
<p>No account? <a href="{{ url_for('register') }}">Register here</a></p>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
"""

register_html = """
<!doctype html>
<title>Register</title>
<h2>Register</h2>
<form method=post>
  <label>Username: <input name=username></label><br>
  <label>Password: <input type=password name=password></label><br>
  <input type=submit value=Register>
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
"""

welcome_html = """
<!doctype html>
<title>Welcome</title>
<h2>Welcome, {{ username }}!</h2>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Area</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
"""

admin_html = """
<!doctype html>
<title>Admin Area</title>
<h2>Admin Area - Registered Users</h2>
<table border=1>
<tr><th>ID</th><th>Username</th><th>Admin</th></tr>
{% for user in users %}
<tr>
  <td>{{ user['id'] }}</td>
  <td>{{ user['username'] }}</td>
  <td>{{ 'Yes' if user['is_admin'] else 'No' }}</td>
</tr>
{% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
"""

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("welcome"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            error = "Missing username or password"
        else:
            with get_db() as db:
                user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
                if user and check_password_hash(user["password"], password):
                    session["user_id"] = user["id"]
                    session["username"] = user["username"]
                    session["is_admin"] = bool(user["is_admin"])
                    return redirect(url_for("welcome"))
                else:
                    error = "Invalid username or password"
    return render_template_string(login_html, error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            error = "Missing username or password"
        else:
            with get_db() as db:
                try:
                    hashed = generate_password_hash(password)
                    db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
                    return redirect(url_for("login"))
                except sqlite3.IntegrityError:
                    error = "Username already taken"
    return render_template_string(register_html, error=error)

@app.route("/welcome")
def welcome():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template_string(welcome_html, username=session["username"], is_admin=session.get("is_admin", False))

@app.route("/admin")
def admin():
    if not session.get("is_admin"):
        return redirect(url_for("login"))
    with get_db() as db:
        users = db.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    return render_template_string(admin_html, users=users)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)