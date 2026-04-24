from flask import Flask, request, redirect, url_for, render_template_string, session
import sqlite3
import os
import hashlib

app = Flask(__name__)
app.secret_key = "supersecretkey123"
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
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

BASE_STYLE = """
<style>
    body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #f4f4f4; }
    h1, h2 { color: #333; }
    input[type=text], input[type=password], input[type=email], textarea {
        width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px;
    }
    button, input[type=submit] {
        background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;
    }
    button:hover, input[type=submit]:hover { background: #45a049; }
    a { color: #4CAF50; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .nav { margin-bottom: 20px; background: #fff; padding: 10px; border-radius: 4px; border: 1px solid #ddd; }
    .nav a { margin-right: 15px; }
    .card { background: #fff; padding: 20px; border-radius: 6px; border: 1px solid #ddd; margin-bottom: 20px; }
    .error { color: red; }
    .success { color: green; }
    label { font-weight: bold; }
    .field-val { background: #eee; padding: 6px 10px; border-radius: 4px; margin-bottom: 10px; }
</style>
"""

def nav_bar():
    if 'user_id' in session:
        return f"""
        <div class='nav'>
            <a href='{url_for("dashboard")}'>Dashboard</a>
            <a href='{url_for("edit_profile")}'>Edit Profile</a>
            <a href='{url_for("view_profile_page")}'>View Profile by ID</a>
            <a href='{url_for("logout")}'>Logout</a>
        </div>
        """
    else:
        return f"""
        <div class='nav'>
            <a href='{url_for("index")}'>Home</a>
            <a href='{url_for("register")}'>Register</a>
            <a href='{url_for("login")}'>Login</a>
        </div>
        """

@app.route("/")
def index():
    return render_template_string(BASE_STYLE + nav_bar() + """
    <div class='card'>
        <h1>Welcome to Profile Manager</h1>
        <p>Create an account to manage your profile information.</p>
        <p><a href='{{ url_for("register") }}'>Register</a> or <a href='{{ url_for("login") }}'>Login</a></p>
    </div>
    """)

@app.route("/register", methods=["GET", "POST"])
def register():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            error = "Username and password are required."
        else:
            try:
                conn = get_db()
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                             (username, hash_password(password)))
                conn.commit()
                conn.close()
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                error = "Username already exists."
    return render_template_string(BASE_STYLE + nav_bar() + """
    <div class='card'>
        <h2>Register</h2>
        {% if error %}<p class='error'>{{ error }}</p>{% endif %}
        <form method='post'>
            <label>Username</label>
            <input type='text' name='username' required>
            <label>Password</label>
            <input type='password' name='password' required>
            <input type='submit' value='Register'>
        </form>
        <p>Already have an account? <a href='{{ url_for("login") }}'>Login</a></p>
    </div>
    """, error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=? AND password=?",
                            (username, hash_password(password))).fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password."
    return render_template_string(BASE_STYLE + nav_bar() + """
    <div class='card'>
        <h2>Login</h2>
        {% if error %}<p class='error'>{{ error }}</p>{% endif %}
        <form method='post'>
            <label>Username</label>
            <input type='text' name='username' required>
            <label>Password</label>
            <input type='password' name='password' required>
            <input type='submit' value='Login'>
        </form>
        <p>Don't have an account? <a href='{{ url_for("register") }}'>Register</a></p>
    </div>
    """, error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for("login"))
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (session['user_id'],)).fetchone()
    conn.close()
    return render_template_string(BASE_STYLE + nav_bar() + """
    <div class='card'>
        <h2>Dashboard</h2>
        <p>Welcome, <strong>{{ username }}</strong>! Your Account ID is <strong>{{ user_id }}</strong>.</p>
        <h3>Your Profile</h3>
        <p><strong>Name:</strong> <span class='field-val'>{{ user['name'] or 'Not set' }}</span></p>
        <p><strong>Email:</strong> <span class='field-val'>{{ user['email'] or 'Not set' }}</span></p>
        <p><strong>Phone:</strong> <span class='field-val'>{{ user['phone'] or 'Not set' }}</span></p>
        <p><strong>Address:</strong> <span class='field-val'>{{ user['address'] or 'Not set' }}</span></p>
        <a href='{{ url_for("edit_profile") }}'><button>Edit Profile</button></a>
    </div>
    """, username=session['username'], user_id=session['user_id'], user=user)

@app.route("/edit_profile", methods=["GET", "POST"])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for("login"))
    success = ""
    error = ""
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (session['user_id'],)).fetchone()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        conn.execute("UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?",
                     (name, email, phone, address, session['user_id']))
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE id=?", (session['user_id'],)).fetchone()
        success = "Profile updated successfully!"
    conn.close()
    return render_template_string(BASE_STYLE + nav_bar() + """
    <div class='card'>
        <h2>Edit Profile</h2>
        {% if success %}<p class='success'>{{ success }}</p>{% endif %}
        {% if error %}<p class='error'>{{ error }}</p>{% endif %}
        <form method='post'>
            <label>Name</label>
            <input type='text' name='name' value='{{ user["name"] or "" }}'>
            <label>Email</label>
            <input type='email' name='email' value='{{ user["email"] or "" }}'>
            <label>Phone Number</label>
            <input type='text' name='phone' value='{{ user["phone"] or "" }}'>
            <label>Address</label>
            <input type='text' name='address' value='{{ user["address"] or "" }}'>
            <input type='submit' value='Update Profile'>
        </form>
    </div>
    """, user=user, success=success, error=error)

@app.route("/view_profile", methods=["GET", "POST"])
def view_profile_page():
    profile = None
    error = ""
    account_id = request.form.get("account_id", "").strip() if request.method == "POST" else ""
    if request.method == "POST":
        if not account_id:
            error = "Please enter an account ID."
        else:
            try:
                aid = int(account_id)
                conn = get_db()
                profile = conn.execute("SELECT id, username, name, email, phone, address FROM users WHERE id=?", (aid,)).fetchone()
                conn.close()
                if not profile:
                    error = f"No account found with ID {aid}."
            except ValueError:
                error = "Account ID must be a number."
    return render_template_string(BASE_STYLE + nav_bar() + """
    <div class='card'>
        <h2>View Profile by Account ID</h2>
        <form method='post'>
            <label>Account ID</label>
            <input type='text' name='account_id' value='{{ account_id }}' placeholder='Enter account ID'>
            <input type='submit' value='Search'>
        </form>
        {% if error %}<p class='error'>{{ error }}</p>{% endif %}
        {% if profile %}
        <hr>
        <h3>Profile Details</h3>
        <p><strong>Account ID:</strong> <span class='field-val'>{{ profile['id'] }}</span></p>
        <p><strong>Username:</strong> <span class='field-val'>{{ profile['username'] }}</span></p>
        <p><strong>Name:</strong> <span class='field-val'>{{ profile['name'] or 'Not set' }}</span></p>
        <p><strong>Email:</strong> <span class='field-val'>{{ profile['email'] or 'Not set' }}</span></p>
        <p><strong>Phone:</strong> <span class='field-val'>{{ profile['phone'] or 'Not set' }}</span></p>
        <p><strong>Address:</strong> <span class='field-val'>{{ profile['address'] or 'Not set' }}</span></p>
        {% endif %}
    </div>
    """, profile=profile, error=error, account_id=account_id)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)