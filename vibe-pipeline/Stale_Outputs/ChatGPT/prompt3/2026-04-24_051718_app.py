from flask import Flask, render_template_string, request, redirect, session, url_for
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_NAME = "users.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )''')
        # Create default admin if not exists
        c.execute("SELECT * FROM users WHERE username = ?", ("admin",))
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      ("admin", generate_password_hash("admin123"), 1))
        conn.commit()

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("welcome"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            return render_template_string(REG_TEMPLATE, error="Username and password required.")
        hashed_password = generate_password_hash(password)
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template_string(REG_TEMPLATE, error="Username already taken.")
    return render_template_string(REG_TEMPLATE, error=None)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            return render_template_string(LOGIN_TEMPLATE, error="Username and password required.")
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            row = c.fetchone()
            if row and check_password_hash(row[0], password):
                session["username"] = username
                return redirect(url_for("welcome"))
            else:
                return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials.")
    return render_template_string(LOGIN_TEMPLATE, error=None)

@app.route("/welcome")
def welcome():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template_string(WELCOME_TEMPLATE, username=session["username"])

@app.route("/admin")
def admin():
    if "username" not in session:
        return redirect(url_for("login"))
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE username = ?", (session["username"],))
        row = c.fetchone()
        if not row or row[0] != 1:
            return "Access denied", 403
        c.execute("SELECT username, is_admin FROM users")
        users = c.fetchall()
    return render_template_string(ADMIN_TEMPLATE, users=users, admin_name=session["username"])

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

REG_TEMPLATE = '''
<!DOCTYPE html>
<html><head><title>Register</title></head><body>
<h2>Register</h2>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method="post">
  Username:<br><input type="text" name="username"><br>
  Password:<br><input type="password" name="password"><br><br>
  <input type="submit" value="Register">
</form>
<p>Already have an account? <a href="{{ url_for('login') }}">Login here</a>.</p>
</body></html>
'''

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html><head><title>Login</title></head><body>
<h2>Login</h2>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
<form method="post">
  Username:<br><input type="text" name="username"><br>
  Password:<br><input type="password" name="password"><br><br>
  <input type="submit" value="Login">
</form>
<p>Don't have an account? <a href="{{ url_for('register') }}">Register here</a>.</p>
</body></html>
'''

WELCOME_TEMPLATE = '''
<!DOCTYPE html>
<html><head><title>Welcome</title></head><body>
<h2>Welcome, {{ username }}!</h2>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
{% if username == 'admin' %}
<p><a href="{{ url_for('admin') }}">Admin Area</a></p>
{% endif %}
</body></html>
'''

ADMIN_TEMPLATE = '''
<!DOCTYPE html>
<html><head><title>Admin Area</title></head><body>
<h2>Admin area - User list (Logged in as {{ admin_name }})</h2>
<table border="1" cellpadding="5" cellspacing="0">
<tr><th>Username</th><th>Role</th></tr>
{% for user, is_admin in users %}
<tr><td>{{ user }}</td><td>{{ 'Admin' if is_admin == 1 else 'User' }}</td></tr>
{% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
</body></html>
'''

if __name__ == "__main__":
    init_db()
    app.run(debug=True)