import sqlite3
from flask import Flask, request, render_template_string, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'internal_portal_key'

def get_db():
    conn = sqlite3.connect('portal.db')
    conn.row_factory = sqlite3.Row
    return conn

def setup_db():
    db = get_db()
    db.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, is_admin INTEGER)')
    try:
        db.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)', ('admin', 'admin123', 1))
    except sqlite3.IntegrityError:
        pass
    db.commit()
    db.close()

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template_string('''
        <h1>Internal Company Portal</h1>
        <p>Welcome, <strong>{{ username }}</strong>!</p>
        <nav>
            <ul>
                {% if admin %}<li><a href="/admin">Admin User Management</a></li>{% endif %}
                <li><a href="/logout">Logout</a></li>
            </ul>
        </nav>
    ''', username=session['user'], admin=session.get('is_admin'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_name = request.form.get('u')
        pass_word = request.form.get('p')
        db = get_db()
        try:
            db.execute('INSERT INTO users (username, password, is_admin) VALUES (?, ?, 0)', (user_name, pass_word))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return 'Username already exists. <a href="/register">Try again</a>'
        finally:
            db.close()
    return '''
        <form method="post">
            <h2>Sign Up</h2>
            Username: <input name="u" required><br><br>
            Password: <input type="password" name="p" required><br><br>
            <input type="submit" value="Register">
        </form>
        <p>Already have an account? <a href="/login">Login here</a></p>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_name = request.form.get('u')
        pass_word = request.form.get('p')
        db = get_db()
        user_record = db.execute('SELECT * FROM users WHERE username = ? AND password = ?', (user_name, pass_word)).fetchone()
        db.close()
        if user_record:
            session['user'] = user_record['username']
            session['is_admin'] = user_record['is_admin']
            return redirect(url_for('home'))
        return 'Invalid credentials. <a href="/login">Retry</a>'
    return '''
        <form method="post">
            <h2>Login</h2>
            Username: <input name="u" required><br><br>
            Password: <input type="password" name="p" required><br><br>
            <input type="submit" value="Login">
        </form>
        <p>New user? <a href="/register">Register here</a></p>
    '''

@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return 'Access Denied: Administrators only.', 403
    db = get_db()
    all_users = db.execute('SELECT id, username, is_admin FROM users').fetchall()
    db.close()
    table_rows = "".join([f"<tr><td>{u['id']}</td><td>{u['username']}</td><td>{'Yes' if u['is_admin'] else 'No'}</td></tr>" for u in all_users])
    return f'''
        <h1>Admin Panel - Registered Users</h1>
        <table border="1" cellpadding="5">
            <tr><th>ID</th><th>Username</th><th>Is Admin</th></tr>
            {table_rows}
        </table>
        <br><a href="/">Back to Home</a>
    '''

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    setup_db()
    app.run(host='0.0.0.0', port=5000)