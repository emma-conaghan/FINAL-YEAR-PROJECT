import sqlite3
from flask import Flask, request, render_template_string, redirect, url_for

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, phone TEXT, address TEXT)')
    conn.commit()
    conn.close()

init_db()

BASE_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>User Management System</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; }
        input[type="text"], input[type="email"], input[type="number"], textarea { width: 300px; padding: 8px; margin: 5px 0; }
        input[type="submit"] { padding: 10px 20px; cursor: pointer; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        <a href="{{ url_for('register') }}">Create Account</a>
    </nav>
    <hr>
    {% block content %}{% endblock %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(BASE_TEMPLATE + """
{% block content %}
    <h1>Welcome</h1>
    <p>Manage user accounts and profiles.</p>
    <div>
        <h3>Search Profile by ID</h3>
        <form action="/view_by_id" method="get">
            <input type="number" name="uid" placeholder="Enter Account ID" required>
            <input type="submit" value="View Profile">
        </form>
    </div>
{% endblock %}
""")

@app.route('/view_by_id')
def view_by_id():
    uid = request.args.get('uid')
    if uid:
        return redirect(url_for('view_profile', user_id=uid))
    return redirect(url_for('index'))

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)', 
                    (name, email, phone, address))
        user_id = cur.lastrowid
        conn.commit()
        conn.close()
        return redirect(url_for('view_profile', user_id=user_id))
    
    return render_template_string(BASE_TEMPLATE + """
{% block content %}
    <h2>Create New Account</h2>
    <form method="post">
        <label>Name:</label><br>
        <input type="text" name="name" required><br>
        <label>Email:</label><br>
        <input type="email" name="email" required><br>
        <label>Phone Number:</label><br>
        <input type="text" name="phone"><br>
        <label>Address:</label><br>
        <textarea name="address" rows="4"></textarea><br>
        <input type="submit" value="Register">
    </form>
{% endblock %}
""")

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user is None:
        return "User with ID " + str(user_id) + " not found.", 404
    
    return render_template_string(BASE_TEMPLATE + """
{% block content %}
    <h2>Profile Details</h2>
    <p><strong>Account ID:</strong> {{ user['id'] }}</p>
    <p><strong>Full Name:</strong> {{ user['name'] }}</p>
    <p><strong>Email Address:</strong> {{ user['email'] }}</p>
    <p><strong>Phone Number:</strong> {{ user['phone'] }}</p>
    <p><strong>Mailing Address:</strong> {{ user['address'] }}</p>
    <br>
    <a href="{{ url_for('edit_profile', user_id=user['id']) }}">Update Profile Information</a>
{% endblock %}
""", user=user)

@app.route('/edit/<int:user_id>', methods=('GET', 'POST'))
def edit_profile(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if user is None:
        conn.close()
        return "User not found", 404

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        
        conn.execute('UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?', 
                     (name, email, phone, address, user_id))
        conn.commit()
        conn.close()
        return redirect(url_for('view_profile', user_id=user_id))
    
    conn.close()
    return render_template_string(BASE_TEMPLATE + """
{% block content %}
    <h2>Update Profile</h2>
    <form method="post">
        <label>Name:</label><br>
        <input type="text" name="name" value="{{ user['name'] }}" required><br>
        <label>Email:</label><br>
        <input type="email" name="email" value="{{ user['email'] }}" required><br>
        <label>Phone Number:</label><br>
        <input type="text" name="phone" value="{{ user['phone'] }}"><br>
        <label>Address:</label><br>
        <textarea name="address" rows="4">{{ user['address'] }}</textarea><br>
        <input type="submit" value="Save Changes">
    </form>
{% endblock %}
""", user=user)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)