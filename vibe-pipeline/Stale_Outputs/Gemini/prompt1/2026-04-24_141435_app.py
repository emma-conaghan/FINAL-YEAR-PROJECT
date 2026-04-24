import sqlite3
from flask import Flask, request, session, redirect, url_for, render_template_string

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-internal-portal'

DATABASE = 'portal.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        try:
            conn.execute("INSERT INTO users (username, password, is_admin) VALUES ('admin', 'admin123', 1)")
        except sqlite3.IntegrityError:
            pass
        conn.commit()

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    html = '''
    <h1>Welcome, {{ username }}!</h1>
    <p>This is the internal company portal.</p>
    {% if is_admin %}
        <p><a href="/admin">Go to Admin Dashboard</a></p>
    {% endif %}
    <p><a href="/logout">Logout</a></p>
    '''
    return render_template_string(html, username=session['username'], is_admin=session.get('is_admin'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        try:
            with sqlite3.connect(DATABASE) as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists. <a href='/register'>Try again</a>"
            
    return '''
    <h1>Register</h1>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <button type="submit">Sign Up</button>
    </form>
    <p>Already have an account? <a href="/login">Login here</a></p>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT is_admin FROM users WHERE username = ? AND password = ?", (username, password))
            user = cursor.fetchone()
            
            if user:
                session['username'] = username
                session['is_admin'] = bool(user[0])
                return redirect(url_for('index'))
            else:
                return "Invalid credentials. <a href='/login'>Try again</a>"
                
    return '''
    <h1>Login</h1>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <button type="submit">Login</button>
    </form>
    <p>No account? <a href="/register">Register here</a></p>
    '''

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return "Access Denied: Admins only.", 403
        
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, is_admin FROM users")
        users = cursor.fetchall()
        
    html = '''
    <h1>Admin Area - Registered Users</h1>
    <table border="1">
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Is Admin</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user[0] }}</td>
            <td>{{ user[1] }}</td>
            <td>{{ 'Yes' if user[2] else 'No' }}</td>
        </tr>
        {% endfor %}
    </table>
    <br>
    <a href="/">Back to Home</a>
    '''
    return render_template_string(html, users=users)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)