import sqlite3
from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

DATABASE = 'database.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                address TEXT
            )
        ''')
        conn.commit()

@app.route('/')
def index():
    return '''
        <h1>User Profile System</h1>
        <ul>
            <li><a href="/signup">Create Account</a></li>
        </ul>
        <form action="/search" method="get">
            View Profile by ID: <input type="number" name="id">
            <button type="submit">Go</button>
        </form>
    '''

@app.route('/search')
def search():
    user_id = request.args.get('id')
    if user_id:
        return redirect(url_for('view_profile', user_id=user_id))
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)',
            (name, email, phone, address)
        )
        new_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return redirect(url_for('view_profile', user_id=new_id))
    
    return '''
        <h1>Create Account</h1>
        <form method="post">
            Name: <input type="text" name="name" required><br>
            Email: <input type="email" name="email" required><br>
            Phone: <input type="text" name="phone"><br>
            Address: <textarea name="address"></textarea><br>
            <button type="submit">Sign Up</button>
        </form>
        <a href="/">Back</a>
    '''

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if user is None:
        return 'User not found', 404
    
    return render_template_string('''
        <h1>Profile: {{ user.name }}</h1>
        <p><strong>ID:</strong> {{ user.id }}</p>
        <p><strong>Email:</strong> {{ user.email }}</p>
        <p><strong>Phone:</strong> {{ user.phone }}</p>
        <p><strong>Address:</strong> {{ user.address }}</p>
        <hr>
        <a href="/update/{{ user.id }}">Update Profile</a> | 
        <a href="/">Home</a>
    ''', user=user)

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update_profile(user_id):
    conn = get_db()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        
        conn.execute(
            'UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?',
            (name, email, phone, address, user_id)
        )
        conn.commit()
        conn.close()
        return redirect(url_for('view_profile', user_id=user_id))
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    
    if user is None:
        return 'User not found', 404

    return render_template_string('''
        <h1>Update Profile</h1>
        <form method="post">
            Name: <input type="text" name="name" value="{{ user.name }}" required><br>
            Email: <input type="email" name="email" value="{{ user.email }}" required><br>
            Phone: <input type="text" name="phone" value="{{ user.phone }}"><br>
            Address: <textarea name="address">{{ user.address }}</textarea><br>
            <button type="submit">Save Changes</button>
        </form>
        <a href="/profile/{{ user.id }}">Cancel</a>
    ''', user=user)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)