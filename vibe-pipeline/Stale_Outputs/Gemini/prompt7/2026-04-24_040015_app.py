import sqlite3
from flask import Flask, request, redirect, render_template_string

app = Flask(__name__)

DATABASE = 'database.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            address TEXT
        )''')
    conn.close()

def query_db(query, args=(), one=False):
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(query, args)
        rv = cur.fetchall()
        conn.commit()
        return (rv[0] if rv else None) if one else rv

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        address = request.form.get('address')
        
        with sqlite3.connect(DATABASE) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)",
                        (name, email, phone, address))
            user_id = cur.lastrowid
            conn.commit()
        return redirect(f'/profile/{user_id}')

    html = '''
    <h1>Create Account</h1>
    <form method="post">
        Name: <input type="text" name="name"><br>
        Email: <input type="email" name="email"><br>
        Phone: <input type="text" name="phone"><br>
        Address: <input type="text" name="address"><br>
        <input type="submit" value="Register">
    </form>
    '''
    return render_template_string(html)

@app.route('/profile/<int:user_id>')
def profile(user_id):
    user = query_db("SELECT * FROM users WHERE id = ?", (user_id,), one=True)
    if not user:
        return "User not found", 404
    
    html = '''
    <h1>Profile Details</h1>
    <p>ID: {{ user['id'] }}</p>
    <p>Name: {{ user['name'] }}</p>
    <p>Email: {{ user['email'] }}</p>
    <p>Phone: {{ user['phone'] }}</p>
    <p>Address: {{ user['address'] }}</p>
    <a href="/update/{{ user['id'] }}">Update Profile</a> | <a href="/">New Account</a>
    '''
    return render_template_string(html, user=user)

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update_profile(user_id):
    user = query_db("SELECT * FROM users WHERE id = ?", (user_id,), one=True)
    if not user:
        return "User not found", 404

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        address = request.form.get('address')
        
        query_db("UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?", 
                 (name, email, phone, address, user_id))
        return redirect(f'/profile/{user_id}')

    html = '''
    <h1>Update Profile</h1>
    <form method="post">
        Name: <input type="text" name="name" value="{{ user['name'] }}"><br>
        Email: <input type="email" name="email" value="{{ user['email'] }}"><br>
        Phone: <input type="text" name="phone" value="{{ user['phone'] }}"><br>
        Address: <input type="text" name="address" value="{{ user['address'] }}"><br>
        <input type="submit" value="Save Changes">
    </form>
    <a href="/profile/{{ user['id'] }}">Cancel</a>
    '''
    return render_template_string(html, user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)