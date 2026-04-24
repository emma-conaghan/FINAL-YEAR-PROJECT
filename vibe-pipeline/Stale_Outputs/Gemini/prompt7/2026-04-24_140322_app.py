import sqlite3
from flask import Flask, request, session, g, redirect, url_for

app = Flask(__name__)
app.secret_key = 'secret_key_123'
DATABASE = 'app.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, name TEXT, email TEXT, phone TEXT, address TEXT)')
        db.commit()

@app.route('/')
def index():
    if 'user_id' in session:
        return f"<h1>Home</h1><p>Logged in as {session['username']} (ID: {session['user_id']})</p><a href='/profile/update'>Update Profile</a> | <a href='/logout'>Logout</a>"
    return "<h1>Home</h1><a href='/login'>Login</a> | <a href='/register'>Register</a>"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        db = get_db()
        try:
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (u, p))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username exists. <a href='/register'>Try again</a>"
    return '<form method="post">Username: <input name="username"><br>Password: <input type="password" name="password"><br><button type="submit">Register</button></form>'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username')
        p = request.form.get('password')
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ? AND password = ?', (u, p)).fetchone()
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
        return "Invalid login. <a href='/login'>Try again</a>"
    return '<form method="post">Username: <input name="username"><br>Password: <input type="password" name="password"><br><button type="submit">Login</button></form>'

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/profile/update', methods=['GET', 'POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    db = get_db()
    if request.method == 'POST':
        db.execute('UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?', 
                   (request.form.get('name'), request.form.get('email'), 
                    request.form.get('phone'), request.form.get('address'), session['user_id']))
        db.commit()
        return "Profile updated! <a href='/'>Home</a>"
    user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    return f'''
    <h1>Update Profile</h1>
    <form method="post">
        Name: <input name="name" value="{user['name'] or ''}"><br>
        Email: <input name="email" value="{user['email'] or ''}"><br>
        Phone: <input name="phone" value="{user['phone'] or ''}"><br>
        Address: <input name="address" value="{user['address'] or ''}"><br>
        <button type="submit">Save</button>
    </form>
    <a href="/">Back</a>
    '''

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if user:
        return f'''
        <h1>User Profile</h1>
        <p><strong>Username:</strong> {user['username']}</p>
        <p><strong>Name:</strong> {user['name'] or 'N/A'}</p>
        <p><strong>Email:</strong> {user['email'] or 'N/A'}</p>
        <p><strong>Phone:</strong> {user['phone'] or 'N/A'}</p>
        <p><strong>Address:</strong> {user['address'] or 'N/A'}</p>
        <a href="/">Back Home</a>
        '''
    return "User not found", 404

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)