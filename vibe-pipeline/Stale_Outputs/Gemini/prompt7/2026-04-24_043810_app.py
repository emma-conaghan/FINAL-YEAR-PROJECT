import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string

app = Flask(__name__)
app.secret_key = 'insecure_secret_key'

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password TEXT, 
                  name TEXT, 
                  email TEXT, 
                  phone TEXT, 
                  address TEXT)''')
    conn.commit()
    conn.close()

init_db()

LAYOUT = """
<!DOCTYPE html>
<html>
<head><title>Profile App</title></head>
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
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p><strong>{{ message }}</strong></p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {{ content | safe }}
</body>
</html>
"""

@app.route('/')
def index():
    content = "<h1>Welcome to the Profile App</h1>"
    if 'user_id' in session:
        content += f"<p>Logged in as ID: {session['user_id']}. <a href='/user/{session['user_id']}'>View Public Profile</a></p>"
    return render_template_string(LAYOUT, content=content)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password, name, email, phone, address) VALUES (?, ?, ?, ?, ?, ?)',
                         (username, password, '', '', '', ''))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
        finally:
            conn.close()
    
    content = """
    <h1>Register</h1>
    <form method="post">
        Username: <input name="username" required><br>
        Password: <input name="password" type="password" required><br>
        <button type="submit">Register</button>
    </form>
    """
    return render_template_string(LAYOUT, content=content)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            return redirect(url_for('profile'))
        return "Invalid credentials."
    
    content = """
    <h1>Login</h1>
    <form method="post">
        Username: <input name="username" required><br>
        Password: <input name="password" type="password" required><br>
        <button type="submit">Login</button>
    </form>
    """
    return render_template_string(LAYOUT, content=content)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    conn = get_db_connection()
    
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        conn.execute('UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?', 
                     (name, email, phone, address, user_id))
        conn.commit()
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()

    content = f"""
    <h1>Update Profile</h1>
    <form method="post">
        Name: <input name="name" value="{user['name']}"><br>
        Email: <input name="email" value="{user['email']}"><br>
        Phone: <input name="phone" value="{user['phone']}"><br>
        Address: <input name="address" value="{user['address']}"><br>
        <button type="submit">Update</button>
    </form>
    <p><a href="/user/{user_id}">View your public profile</a></p>
    """
    return render_template_string(LAYOUT, content=content)

@app.route('/user/<int:user_id>')
def view_user(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if not user:
        return "User not found", 404
    
    content = f"""
    <h1>Profile Details</h1>
    <p><strong>Account ID:</strong> {user['id']}</p>
    <p><strong>Username:</strong> {user['username']}</p>
    <p><strong>Name:</strong> {user['name']}</p>
    <p><strong>Email:</strong> {user['email']}</p>
    <p><strong>Phone:</strong> {user['phone']}</p>
    <p><strong>Address:</strong> {user['address']}</p>
    """
    return render_template_string(LAYOUT, content=content)

if __name__ == '__main__':
    app.run(debug=True)