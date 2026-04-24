import sqlite3
from flask import Flask, request, render_template_string, redirect

app = Flask(__name__)
DB_PATH = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, phone TEXT, address TEXT)')
    conn.commit()
    conn.close()

init_db()

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>User Profile App</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; line-height: 1.6; background: #f4f4f4; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        input { margin-bottom: 10px; width: 100%; padding: 8px; box-sizing: border-box; display: block; }
        button { padding: 10px 15px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; }
        nav { margin-bottom: 20px; }
        label { font-weight: bold; display: block; margin-bottom: 5px; }
    </style>
</head>
<body>
    <nav><a href="/">Home / Register</a></nav>
    <div class="card">
        {{ content | safe }}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    form_html = """
    <h1>Create Account</h1>
    <form action="/register" method="POST">
        <label>Name</label><input name="name" required>
        <label>Email</label><input name="email" type="email" required>
        <label>Phone</label><input name="phone">
        <label>Address</label><input name="address">
        <button type="submit">Register</button>
    </form>
    """
    return render_template_string(BASE_TEMPLATE, content=form_html)

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)', (name, email, phone, address))
    new_id = cur.lastrowid
    conn.commit()
    conn.close()
    return redirect(f'/profile/{new_id}')

@app.route('/profile/<int:user_id>')
def profile(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if user is None:
        return "User not found", 404
    
    display_html = f"""
    <h1>Profile Details</h1>
    <p><strong>Account ID:</strong> {user['id']}</p>
    <p><strong>Name:</strong> {user['name']}</p>
    <p><strong>Email:</strong> {user['email']}</p>
    <p><strong>Phone:</strong> {user['phone']}</p>
    <p><strong>Address:</strong> {user['address']}</p>
    <hr>
    <a href="/edit/{user['id']}"><button style="background:#007bff;">Edit Profile Information</button></a>
    """
    return render_template_string(BASE_TEMPLATE, content=display_html)

@app.route('/edit/<int:user_id>')
def edit(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if user is None:
        return "User not found", 404
    
    edit_html = f"""
    <h1>Update Profile</h1>
    <form action="/update/{user['id']}" method="POST">
        <label>Name</label><input name="name" value="{user['name']}">
        <label>Email</label><input name="email" type="email" value="{user['email']}">
        <label>Phone</label><input name="phone" value="{user['phone']}">
        <label>Address</label><input name="address" value="{user['address']}">
        <button type="submit">Save Changes</button>
    </form>
    """
    return render_template_string(BASE_TEMPLATE, content=edit_html)

@app.route('/update/<int:user_id>', methods=['POST'])
def update(user_id):
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    address = request.form.get('address')
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?', (name, email, phone, address, user_id))
    conn.commit()
    conn.close()
    return redirect(f'/profile/{user_id}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)