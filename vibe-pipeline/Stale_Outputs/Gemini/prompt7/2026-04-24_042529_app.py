import sqlite3
from flask import Flask, request, redirect, session, url_for, render_template_string

app = Flask(__name__)
app.secret_key = 'temporary-secret-key-for-development'
DATABASE = 'app.db'

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )''')
        db.commit()

init_db()

HTML_LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Profile Manager</title>
    <style>
        body { font-family: sans-serif; margin: 20px; line-height: 1.6; }
        nav { margin-bottom: 20px; }
        .form-group { margin-bottom: 10px; }
        label { display: inline-block; width: 100px; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a> | 
        {% if session.get('user_id') %}
            <a href="/profile">My Profile</a> | 
            <a href="/logout">Logout</a>
        {% else %}
            <a href="/login">Login</a> | 
            <a href="/register">Register</a>
        {% endif %}
    </nav>
    <hr>
    {% block content %}{% endblock %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
    <h1>Welcome to Profile Manager</h1>
    <p>Search for a user profile by Account ID:</p>
    <form action="/view_user" method="get">
        <input type="number" name="id" placeholder="Account ID" required>
        <input type="submit" value="View Profile">
    </form>
    {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            db = get_db()
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists. Please choose another."
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
    <h1>Create Account</h1>
    <form method="post">
        <div class="form-group"><label>Username:</label><input name="username" required></div>
        <div class="form-group"><label>Password:</label><input type="password" name="password" required></div>
        <input type="submit" value="Register">
    </form>
    {% endblock %}
    """)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
        if user:
            session['user_id'] = user['id']
            return redirect(url_for('profile'))
        return "Invalid username or password."
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
    <h1>Login</h1>
    <form method="post">
        <div class="form-group"><label>Username:</label><input name="username" required></div>
        <div class="form-group"><label>Password:</label><input type="password" name="password" required></div>
        <input type="submit" value="Login">
    </form>
    {% endblock %}
    """)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    db = get_db()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        db.execute("UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?", 
                   (name, email, phone, address, session['user_id']))
        db.commit()
    
    user = db.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
    <h1>Update Your Profile (Account ID: {{user['id']}})</h1>
    <form method="post">
        <div class="form-group"><label>Name:</label><input name="name" value="{{user['name'] or ''}}"></div>
        <div class="form-group"><label>Email:</label><input name="email" value="{{user['email'] or ''}}"></div>
        <div class="form-group"><label>Phone:</label><input name="phone" value="{{user['phone'] or ''}}"></div>
        <div class="form-group"><label>Address:</label><input name="address" value="{{user['address'] or ''}}"></div>
        <input type="submit" value="Save Changes">
    </form>
    {% endblock %}
    """, user=user)

@app.route('/view_user')
def view_user():
    user_id = request.args.get('id')
    if not user_id:
        return redirect(url_for('index'))
    
    db = get_db()
    user = db.execute("SELECT id, username, name, email, phone, address FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        return "User not found."
        
    return render_template_string(HTML_LAYOUT + """
    {% block content %}
    <h1>Profile Details</h1>
    <p><strong>Account ID:</strong> {{user['id']}}</p>
    <p><strong>Username:</strong> {{user['username']}}</p>
    <p><strong>Name:</strong> {{user['name'] or 'N/A'}}</p>
    <p><strong>Email:</strong> {{user['email'] or 'N/A'}}</p>
    <p><strong>Phone:</strong> {{user['phone'] or 'N/A'}}</p>
    <p><strong>Address:</strong> {{user['address'] or 'N/A'}}</p>
    <hr>
    <a href="/">Back to Search</a>
    {% endblock %}
    """, user=user)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)