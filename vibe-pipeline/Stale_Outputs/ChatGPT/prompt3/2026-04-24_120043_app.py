from flask import Flask, request, redirect, render_template_string, session, url_for
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

DB_NAME = "users.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        # Create a default admin if not exists
        c.execute("SELECT * FROM users WHERE username=?", ("admin",))
        if not c.fetchone():
            admin_password = hashlib.sha256("admin".encode()).hexdigest()
            c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", 
                      ("admin", admin_password, 1))
            conn.commit()

def get_user(username):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT id, username, password, is_admin FROM users WHERE username=?", (username,))
        return c.fetchone()

def add_user(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False

def get_all_users():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT username, is_admin FROM users")
        return c.fetchall()

init_db()

login_template = '''
<h2>Login</h2>
<form method="post">
  Username: <input name="username" required><br>
  Password: <input name="password" type="password" required><br>
  <input type="submit" value="Login">
</form>
<p>New user? <a href="{{ url_for('register') }}">Register here</a></p>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
'''

register_template = '''
<h2>Register</h2>
<form method="post">
  Username: <input name="username" required><br>
  Password: <input name="password" type="password" required><br>
  <input type="submit" value="Register">
</form>
<p>Already registered? <a href="{{ url_for('login') }}">Login here</a></p>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
{% if message %}<p style="color:green">{{ message }}</p>{% endif %}
'''

welcome_template = '''
<h2>Welcome {{ username }}!</h2>
<p>You are logged in.</p>
{% if is_admin %}
<p><a href="{{ url_for('admin') }}">Go to Admin Panel</a></p>
{% endif %}
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

admin_template = '''
<h2>Admin Panel</h2>
<p>List of registered users:</p>
<table border="1">
  <tr><th>Username</th><th>Is Admin</th></tr>
  {% for user, admin in users %}
    <tr><td>{{ user }}</td><td>{{ "Yes" if admin else "No" }}</td></tr>
  {% endfor %}
</table>
<p><a href="{{ url_for('welcome') }}">Back to Welcome</a></p>
<p><a href="{{ url_for('logout') }}">Logout</a></p>
'''

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("welcome"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = get_user(username)
        if user:
            hashed = hashlib.sha256(password.encode()).hexdigest()
            if hashed == user[2]:
                session["username"] = user[1]
                session["is_admin"] = bool(user[3])
                return redirect(url_for("welcome"))
            else:
                error = "Invalid password"
        else:
            error = "Invalid username"
    return render_template_string(login_template, error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    message = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if add_user(username, password):
            message = "User registered successfully. You can now log in."
        else:
            error = "Username already exists."
    return render_template_string(register_template, error=error, message=message)

@app.route("/welcome")
def welcome():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template_string(welcome_template, username=session["username"], is_admin=session.get("is_admin", False))

@app.route("/admin")
def admin():
    if "username" not in session or not session.get("is_admin", False):
        return redirect(url_for("login"))
    users = get_all_users()
    return render_template_string(admin_template, users=users)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)