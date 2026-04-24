from flask import Flask, render_template_string, request, redirect, url_for, session, flash
import sqlite3
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'

DB_PATH = 'portal.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Create default admin user
    admin_password = hashlib.md5('admin123'.encode()).hexdigest()
    try:
        cursor.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
                      ('admin', admin_password, 1))
    except sqlite3.IntegrityError:
        pass
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

BASE_STYLE = '''
<style>
    body { font-family: Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
    .container { max-width: 400px; margin: 80px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .wide-container { max-width: 800px; margin: 80px auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1, h2 { color: #333; text-align: center; }
    input[type=text], input[type=password] { width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
    button, .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-size: 16px; text-decoration: none; display: inline-block; text-align: center; margin-top: 10px; }
    button:hover, .btn:hover { background: #45a049; }
    .btn-danger { background: #e74c3c; }
    .btn-danger:hover { background: #c0392b; }
    .btn-secondary { background: #3498db; }
    .btn-secondary:hover { background: #2980b9; }
    .btn-small { width: auto; padding: 5px 15px; font-size: 14px; }
    .flash { padding: 10px; margin: 10px 0; border-radius: 4px; text-align: center; }
    .flash-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .flash-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .flash-info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
    .nav { background: #333; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }
    .nav a { color: white; text-decoration: none; margin-left: 15px; }
    .nav a:hover { color: #4CAF50; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #4CAF50; color: white; }
    tr:hover { background: #f5f5f5; }
    .links { text-align: center; margin-top: 15px; }
    .links a { color: #3498db; }
    .welcome-box { text-align: center; padding: 20px; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }
    .badge-admin { background: #e74c3c; color: white; }
    .badge-user { background: #3498db; color: white; }
</style>
'''

LOGIN_TEMPLATE = BASE_STYLE + '''
<div class="container">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Login</h3>
    {% for category, message in get_flashed_messages(with_categories=true) %}
        <div class="flash flash-{{ category }}">{{ message }}</div>
    {% endfor %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    <div class="links">
        Don't have an account? <a href="/register">Register here</a>
    </div>
</div>
'''

REGISTER_TEMPLATE = BASE_STYLE + '''
<div class="container">
    <h2>🏢 Company Portal</h2>
    <h3 style="text-align:center; color:#666;">Register</h3>
    {% for category, message in get_flashed_messages(with_categories=true) %}
        <div class="flash flash-{{ category }}">{{ message }}</div>
    {% endfor %}
    <form method="POST">
        <input type="text" name="username" placeholder="Username" required>
        <input type="password" name="password" placeholder="Password" required>
        <input type="password" name="confirm_password" placeholder="Confirm Password" required>
        <button type="submit">Register</button>
    </form>
    <div class="links">
        Already have an account? <a href="/">Login here</a>
    </div>
</div>
'''

WELCOME_TEMPLATE = BASE_STYLE + '''
<div class="nav">
    <span style="color:white; font-weight:bold;">🏢 Company Portal</span>
    <div>
        {% if session.get('is_admin') %}
            <a href="/admin">Admin Panel</a>
        {% endif %}
        <a href="/logout">Logout</a>
    </div>
</div>
<div class="wide-container">
    <div class="welcome-box">
        <h1>Welcome, {{ username }}! 👋</h1>
        {% if is_admin %}
            <p><span class="badge badge-admin">Administrator</span></p>
        {% else %}
            <p><span class="badge badge-user">Employee</span></p>
        {% endif %}
        <p style="color:#666; font-size:18px;">You have successfully logged into the Company Portal.</p>
        <p style="color:#888;">Use the navigation above to access available features.</p>
        {% if is_admin %}
            <a href="/admin" class="btn btn-secondary" style="width:auto; display:inline-block;">Go to Admin Panel</a>
        {% endif %}
        <br><br>
        <a href="/logout" class="btn btn-danger" style="width:auto; display:inline-block;">Logout</a>
    </div>
</div>
'''

ADMIN_TEMPLATE = BASE_STYLE + '''
<div class="nav">
    <span style="color:white; font-weight:bold;">🏢 Company Portal - Admin</span>
    <div>
        <a href="/welcome">Dashboard</a>
        <a href="/logout">Logout</a>
    </div>
</div>
<div class="wide-container">
    <h2>Admin Panel - All Users</h2>
    {% for category, message in get_flashed_messages(with_categories=true) %}
        <div class="flash flash-{{ category }}">{{ message }}</div>
    {% endfor %}
    <p>Total users: <strong>{{ users|length }}</strong></p>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Role</th>
            <th>Created At</th>
            <th>Actions</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user['id'] }}</td>
            <td>{{ user['username'] }}</td>
            <td>
                {% if user['is_admin'] %}
                    <span class="badge badge-admin">Admin</span>
                {% else %}
                    <span class="badge badge-user">User</span>
                {% endif %}
            </td>
            <td>{{ user['created_at'] }}</td>
            <td>
                {% if user['username'] != 'admin' and user['username'] != session.get('username') %}
                    <form method="POST" action="/admin/delete/{{ user['id'] }}" style="display:inline;">
                        <button type="submit" class="btn btn-danger btn-small" onclick="return confirm('Delete this user?')">Delete</button>
                    </form>
                {% else %}
                    <span style="color:#999; font-size:12px;">Protected</span>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
</div>
'''

@app.route('/', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('welcome'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Please fill in all fields.', 'error')
            return render_template_string(LOGIN_TEMPLATE)
        
        conn = get_db()
        cursor = conn.cursor()
        hashed = hash_password(password)
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['username'] = user['username']
            session['user_id'] = user['id']
            session['is_admin'] = bool(user['is_admin'])
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('welcome'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'username' in session:
        return redirect(url_for('welcome'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if len(password) < 4:
            flash('Password must be at least 4 characters long.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template_string(REGISTER_TEMPLATE)
        
        conn = get_db()
        cursor = conn.cursor()
        try:
            hashed = hash_password(password)
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.', 'error')
        finally:
            conn.close()
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/welcome')
def welcome():
    if 'username' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    return render_template_string(WELCOME_TEMPLATE,
                                  username=session['username'],
                                  is_admin=session.get('is_admin', False))

@app.route('/admin')
def admin():
    if 'username' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    if not session.get('is_admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('welcome'))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    
    return render_template_string(ADMIN_TEMPLATE, users=users)

@app.route('/admin/delete/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'username' not in session or not session.get('is_admin'):
        flash('Access denied.', 'error')
        return redirect(url_for('login'))
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if user and user['username'] != 'admin' and user['username'] != session['username']:
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        flash(f'User "{user["username"]}" has been deleted.', 'success')
    else:
        flash('Cannot delete this user.', 'error')
    
    conn.close()
    return redirect(url_for('admin'))

@app.route('/logout')
def logout():
    username = session.get('username', '')
    session.clear()
    flash(f'You have been logged out, {username}.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)