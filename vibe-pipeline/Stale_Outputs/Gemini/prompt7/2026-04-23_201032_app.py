import sqlite3
from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

DATABASE = 'database.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      email TEXT,
                      phone TEXT,
                      address TEXT)''')
    conn.close()

@app.route('/')
def index():
    return render_template_string('''
        <h1>User Management</h1>
        <h2>Create Account</h2>
        <form action="/register" method="post">
            Name: <input type="text" name="name" required><br>
            Email: <input type="email" name="email" required><br>
            Phone: <input type="text" name="phone"><br>
            Address: <input type="text" name="address"><br>
            <input type="submit" value="Register">
        </form>
        <hr>
        <h2>Search Profile</h2>
        <form action="/view_search" method="get">
            Account ID: <input type="number" name="user_id" required>
            <input type="submit" value="View Profile">
        </form>
    ''')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
                       (name, email, phone, address))
        new_id = cursor.lastrowid
    return redirect(url_for('view_profile', user_id=new_id))

@app.route('/view_search')
def view_search():
    target_id = request.args.get('user_id')
    return redirect(url_for('view_profile', user_id=target_id))

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT name, email, phone, address FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
    if user_data:
        return render_template_string('''
            <h1>Profile Details</h1>
            <p><strong>Account ID:</strong> {{ uid }}</p>
            <p><strong>Name:</strong> {{ data[0] }}</p>
            <p><strong>Email:</strong> {{ data[1] }}</p>
            <p><strong>Phone:</strong> {{ data[2] }}</p>
            <p><strong>Address:</strong> {{ data[3] }}</p>
            <a href="/edit/{{ uid }}">Edit This Profile</a><br><br>
            <a href="/">Back to Home</a>
        ''', data=user_data, uid=user_id)
    return "User not found. <a href='/'>Go back</a>", 404

@app.route('/edit/<int:user_id>')
def edit_profile(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT name, email, phone, address FROM users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()
    if user_data:
        return render_template_string('''
            <h1>Edit Profile</h1>
            <form action="/update/{{ uid }}" method="post">
                Name: <input type="text" name="name" value="{{ data[0] }}" required><br>
                Email: <input type="email" name="email" value="{{ data[1] }}" required><br>
                Phone: <input type="text" name="phone" value="{{ data[2] }}"><br>
                Address: <input type="text" name="address" value="{{ data[3] }}"><br>
                <input type="submit" value="Save Changes">
            </form>
            <br>
            <a href="/profile/{{ uid }}">Cancel</a>
        ''', data=user_data, uid=user_id)
    return "User not found", 404

@app.route('/update/<int:user_id>', methods=['POST'])
def update_profile(user_id):
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?',
                     (name, email, phone, address, user_id))
    return redirect(url_for('view_profile', user_id=user_id))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)