import sqlite3
from flask import Flask, request, redirect, render_template_string, url_for

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT, email TEXT, phone TEXT, address TEXT)''')
    conn.commit()
    conn.close()

def layout(content):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>User Profile System</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; line-height: 1.6; }}
            input, textarea {{ display: block; margin-bottom: 10px; width: 300px; padding: 5px; }}
            .nav {{ margin-bottom: 20px; }}
            .profile-box {{ border: 1px solid #ccc; padding: 20px; width: 400px; }}
        </style>
    </head>
    <body>
        <div class="nav">
            <a href="/">Home</a> | <a href="/register">Create Account</a>
        </div>
        {content}
    </body>
    </html>
    """

@app.route('/')
def index():
    content = """
    <h1>Profile Management</h1>
    <form action="/search" method="get">
        <label>Find profile by Account ID:</label>
        <input type="number" name="user_id" required>
        <button type="submit">View Profile</button>
    </form>
    """
    return render_template_string(layout(content))

@app.route('/search')
def search():
    user_id = request.args.get('user_id')
    if user_id:
        return redirect(url_for('view_profile', user_id=user_id))
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)",
                    (name, email, phone, address))
        new_id = cur.lastrowid
        conn.commit()
        conn.close()
        return redirect(url_for('view_profile', user_id=new_id))

    content = """
    <h1>Create Account</h1>
    <form method="post">
        <input type="text" name="name" placeholder="Full Name" required>
        <input type="email" name="email" placeholder="Email" required>
        <input type="text" name="phone" placeholder="Phone Number" required>
        <textarea name="address" placeholder="Address" required></textarea>
        <button type="submit">Register</button>
    </form>
    """
    return render_template_string(layout(content))

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if user is None:
        return render_template_string(layout("<h1>User Not Found</h1>"))
    
    content = f"""
    <h1>Profile Details</h1>
    <div class="profile-box">
        <p><strong>Account ID:</strong> {user['id']}</p>
        <p><strong>Name:</strong> {user['name']}</p>
        <p><strong>Email:</strong> {user['email']}</p>
        <p><strong>Phone:</strong> {user['phone']}</p>
        <p><strong>Address:</strong> {user['address']}</p>
        <a href="/update/{user['id']}">Edit Profile Information</a>
    </div>
    """
    return render_template_string(layout(content))

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update_profile(user_id):
    conn = get_db_connection()
    
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
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if user is None:
        return render_template_string(layout("<h1>User Not Found</h1>"))
    
    content = f"""
    <h1>Update Profile</h1>
    <form method="post">
        <label>Name:</label>
        <input type="text" name="name" value="{user['name']}" required>
        <label>Email:</label>
        <input type="email" name="email" value="{user['email']}" required>
        <label>Phone:</label>
        <input type="text" name="phone" value="{user['phone']}" required>
        <label>Address:</label>
        <textarea name="address" required>{user['address']}</textarea>
        <button type="submit">Save Changes</button>
    </form>
    """
    return render_template_string(layout(content))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)