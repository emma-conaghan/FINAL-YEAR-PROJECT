import sqlite3
from flask import Flask, request, redirect, url_for

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, phone TEXT, address TEXT)')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Register New Account</h1>
        <form action="/register" method="post">
            <label>Name:</label><br><input type="text" name="name" required><br>
            <label>Email:</label><br><input type="email" name="email" required><br>
            <label>Phone:</label><br><input type="text" name="phone"><br>
            <label>Address:</label><br><textarea name="address"></textarea><br>
            <button type="submit">Create Account</button>
        </form>
        <hr>
        <form action="/search" method="get">
            <label>Lookup Profile by ID:</label>
            <input type="number" name="uid">
            <button type="submit">View</button>
        </form>
    </body>
    </html>
    '''

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')
    
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute('INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)', (name, email, phone, address))
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return redirect(url_for('view_profile', user_id=new_id))

@app.route('/search')
def search():
    uid = request.args.get('uid')
    if uid:
        return redirect(url_for('view_profile', user_id=uid))
    return redirect(url_for('index'))

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute('SELECT id, name, email, phone, address FROM users WHERE id = ?', (user_id,))
    user = cur.fetchone()
    conn.close()
    
    if not user:
        return "User not found. <a href='/'>Go back</a>"
    
    return f'''
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Profile Details</h1>
        <p><strong>Account ID:</strong> {user[0]}</p>
        <p><strong>Name:</strong> {user[1]}</p>
        <p><strong>Email:</strong> {user[2]}</p>
        <p><strong>Phone:</strong> {user[3]}</p>
        <p><strong>Address:</strong> {user[4]}</p>
        <hr>
        <a href="/edit/{user[0]}">Edit Profile</a> | <a href="/">Home</a>
    </body>
    </html>
    '''

@app.route('/edit/<int:user_id>', methods=['GET', 'POST'])
def edit_profile(user_id):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        address = request.form.get('address')
        
        cur.execute('UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?', (name, email, phone, address, user_id))
        conn.commit()
        conn.close()
        return redirect(url_for('view_profile', user_id=user_id))
    
    cur.execute('SELECT name, email, phone, address FROM users WHERE id = ?', (user_id,))
    user = cur.fetchone()
    conn.close()
    
    if not user:
        return "User not found."

    return f'''
    <!DOCTYPE html>
    <html>
    <body>
        <h1>Update Profile</h1>
        <form method="post">
            <label>Name:</label><br><input type="text" name="name" value="{user[0]}"><br>
            <label>Email:</label><br><input type="email" name="email" value="{user[1]}"><br>
            <label>Phone:</label><br><input type="text" name="phone" value="{user[2]}"><br>
            <label>Address:</label><br><textarea name="address">{user[3]}</textarea><br>
            <button type="submit">Save Changes</button>
        </form>
        <br>
        <a href="/profile/{user_id}">Cancel</a>
    </body>
    </html>
    '''

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)