import sqlite3
from flask import Flask, request, redirect, render_template_string

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, phone TEXT, address TEXT)')
    conn.commit()
    conn.close()

init_db()

FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        form { display: flex; flex-direction: column; width: 300px; }
        input { margin-bottom: 10px; padding: 8px; }
        label { font-weight: bold; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <form method="POST">
        <label>Name</label>
        <input type="text" name="name" value="{{ user[1] if user else '' }}" required>
        <label>Email</label>
        <input type="email" name="email" value="{{ user[2] if user else '' }}" required>
        <label>Phone</label>
        <input type="text" name="phone" value="{{ user[3] if user else '' }}">
        <label>Address</label>
        <input type="text" name="address" value="{{ user[4] if user else '' }}">
        <button type="submit">Save Profile</button>
    </form>
    <p><a href="/">Back to Registration</a></p>
</body>
</html>
"""

VIEW_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Profile View</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        .profile { border: 1px solid #ccc; padding: 20px; border-radius: 8px; max-width: 400px; }
    </style>
</head>
<body>
    <h1>Profile Details</h1>
    <div class="profile">
        <p><strong>Account ID:</strong> {{ user[0] }}</p>
        <p><strong>Name:</strong> {{ user[1] }}</p>
        <p><strong>Email:</strong> {{ user[2] }}</p>
        <p><strong>Phone:</strong> {{ user[3] }}</p>
        <p><strong>Address:</strong> {{ user[4] }}</p>
    </div>
    <br>
    <a href="/update/{{ user[0] }}">Edit Profile</a> | 
    <a href="/">Register New User</a>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        address = request.form.get('address')
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)', (name, email, phone, address))
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return redirect(f'/view/{new_id}')
    return render_template_string(FORM_HTML, title="Create Account", user=None)

@app.route('/update/<int:uid>', methods=['GET', 'POST'])
def update_profile(uid):
    conn = sqlite3.connect('database.db')
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        address = request.form.get('address')
        conn.execute('UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?', (name, email, phone, address, uid))
        conn.commit()
        conn.close()
        return redirect(f'/view/{uid}')
    
    user_data = conn.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
    conn.close()
    if not user_data:
        return "Account not found", 404
    return render_template_string(FORM_HTML, title="Update Profile Information", user=user_data)

@app.route('/view/<int:uid>')
def view_profile(uid):
    conn = sqlite3.connect('database.db')
    user_data = conn.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
    conn.close()
    if user_data:
        return render_template_string(VIEW_HTML, user=user_data)
    return "Account not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)