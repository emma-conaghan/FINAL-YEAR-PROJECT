import sqlite3
from flask import Flask, request, redirect, url_for, session, render_template_string

app = Flask(__name__)
app.secret_key = 'supersecretkey'

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    if 'user_id' in session:
        return f'<h1>Welcome</h1><a href="/profile/update">Update Profile</a> | <a href="/profile/{session["user_id"]}">View Profile</a> | <a href="/logout">Logout</a>'
    return '<h1>Welcome</h1><a href="/register">Register</a> | <a href="/login">Login</a>'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
        finally:
            conn.close()
    return '''
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <button type="submit">Register</button>
        </form>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            return redirect(url_for('index'))
        return "Invalid credentials."
    return '''
        <form method="post">
            Username: <input type="text" name="username" required><br>
            Password: <input type="password" name="password" required><br>
            <button type="submit">Login</button>
        </form>
    '''

@app.route('/profile/update', methods=['GET', 'POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        address = request.form.get('address')
        cursor.execute('''
            UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?
        ''', (name, email, phone, address, session['user_id']))
        conn.commit()
        conn.close()
        return redirect(url_for('view_profile', account_id=session['user_id']))
    
    cursor.execute("SELECT name, email, phone, address FROM users WHERE id = ?", (session['user_id'],))
    user_data = cursor.fetchone()
    conn.close()
    
    return render_template_string('''
        <h1>Update Profile</h1>
        <form method="post">
            Name: <input type="text" name="name" value="{{u[0] or ''}}"><br>
            Email: <input type="text" name="email" value="{{u[1] or ''}}"><br>
            Phone: <input type="text" name="phone" value="{{u[2] or ''}}"><br>
            Address: <input type="text" name="address" value="{{u[3] or ''}}"><br>
            <button type="submit">Save Changes</button>
        </form>
    ''', u=user_data)

@app.route('/profile/<int:account_id>')
def view_profile(account_id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, name, email, phone, address FROM users WHERE id = ?", (account_id,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return "User not found", 404
    
    return render_template_string('''
        <h1>Profile Details</h1>
        <p>Account ID: {{aid}}</p>
        <p>Username: {{u[0]}}</p>
        <p>Name: {{u[1] or 'Not set'}}</p>
        <p>Email: {{u[2] or 'Not set'}}</p>
        <p>Phone: {{u[3] or 'Not set'}}</p>
        <p>Address: {{u[4] or 'Not set'}}</p>
        <a href="/">Back Home</a>
    ''', u=user, aid=account_id)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)