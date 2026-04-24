import sqlite3
from flask import Flask, request, render_template_string, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'internal_portal_secret_key'

def init_db():
    conn = sqlite3.connect('portal.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE NOT NULL, 
                  password TEXT NOT NULL, 
                  role TEXT NOT NULL)''')
    try:
        # Default admin account
        admin_pass = generate_password_hash('admin123')
        conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                     ('admin', admin_pass, 'admin'))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h1>Welcome, {{ username }}!</h1>
        <p>This is the internal company portal.</p>
        <nav>
            {% if role == 'admin' %}
                <a href="{{ url_for('admin_panel') }}">Admin Panel</a> |
            {% endif %}
            <a href="{{ url_for('logout') }}">Logout</a>
        </nav>
    ''', username=session['username'], role=session['role'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_pw = generate_password_hash(password)
        
        db = get_db()
        try:
            db.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                       (username, hashed_pw, 'user'))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists. <a href='/register'>Try again</a>"
        finally:
            db.close()
            
    return '''
        <h2>Register</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <button type="submit">Sign Up</button>
        </form>
        <a href="/login">Already have an account? Login</a>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        db.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        return "Invalid credentials. <a href='/login'>Try again</a>"
        
    return '''
        <h2>Login</h2>
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <button type="submit">Login</button>
        </form>
        <a href="/register">Don't have an account? Register</a>
    '''

@app.route('/admin')
def admin_panel():
    if session.get('role') != 'admin':
        return "Access denied: Admins only.", 403
    
    db = get_db()
    users = db.execute("SELECT id, username, role FROM users").fetchall()
    db.close()
    
    return render_template_string('''
        <h2>Admin Panel - Registered Users</h2>
        <table border="1">
            <tr><th>ID</th><th>Username</th><th>Role</th></tr>
            {% for user in users %}
            <tr>
                <td>{{ user.id }}</td>
                <td>{{ user.username }}</td>
                <td>{{ user.role }}</td>
            </tr>
            {% endfor %}
        </table>
        <br><a href="/">Back to Home</a>
    ''', users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)