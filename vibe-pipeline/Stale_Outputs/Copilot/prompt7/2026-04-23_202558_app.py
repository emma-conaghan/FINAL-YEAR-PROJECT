from flask import Flask, render_template_string, request, redirect, url_for, session
import sqlite3

app = Flask(__name__)
app.secret_key = 'secret'

def init_db():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT,
        email TEXT,
        phone TEXT,
        address TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

layout = '''
<!doctype html>
<html>
<head>
<title>{{ title }}</title>
</head>
<body>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul>{% for message in messages %}
      <li>{{ message }}</li>
    {% endfor %}</ul>
  {% endif %}
{% endwith %}
{{ content }}
</body>
</html>
'''

signup_template = '''
{% set title = "Sign Up" %}
{% set content %}
<h2>Sign Up</h2>
<form method="post">
  Username: <input type="text" name="username" required><br>
  Password: <input type="password" name="password" required><br>
  <input type="submit" value="Sign Up">
</form>
<a href="{{ url_for('login') }}">Login</a>
{% endset %}
''' + layout

login_template = '''
{% set title = "Login" %}
{% set content %}
<h2>Login</h2>
<form method="post">
  Username: <input type="text" name="username" required><br>
  Password: <input type="password" name="password" required><br>
  <input type="submit" value="Login">
</form>
<a href="{{ url_for('signup') }}">Sign Up</a>
{% endset %}
''' + layout

profile_template = '''
{% set title = "Update Profile" %}
{% set content %}
<h2>Update Profile</h2>
<form method="post">
  Name: <input type="text" name="name" value="{{ user['name']|default('') }}"><br>
  Email: <input type="email" name="email" value="{{ user['email']|default('') }}"><br>
  Phone: <input type="text" name="phone" value="{{ user['phone']|default('') }}"><br>
  Address: <input type="text" name="address" value="{{ user['address']|default('') }}"><br>
  <input type="submit" value="Update">
</form>
<a href="{{ url_for('profile_view', user_id=user['id']) }}">View Profile</a><br>
<a href="{{ url_for('logout') }}">Logout</a>
{% endset %}
''' + layout

view_profile_template = '''
{% set title = "View Profile" %}
{% set content %}
<h2>Profile Details</h2>
{% if user %}
Name: {{ user['name'] }}<br>
Email: {{ user['email'] }}<br>
Phone: {{ user['phone'] }}<br>
Address: {{ user['address'] }}<br>
{% else %}
User not found.<br>
{% endif %}
<a href="{{ url_for('login') }}">Login</a>
{% endset %}
''' + layout

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('profile'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('app.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template_string(signup_template, error='Username already exists.')
    return render_template_string(signup_template)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('app.db')
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            return redirect(url_for('profile'))
        else:
            return render_template_string(login_template, error='Invalid credentials.')
    return render_template_string(login_template)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        c.execute("UPDATE users SET name = ?, email = ?, phone = ?, address = ? WHERE id = ?", (name, email, phone, address, user_id))
        conn.commit()
    c.execute("SELECT id, username, name, email, phone, address FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        user = {'id': row[0], 'username': row[1], 'name': row[2], 'email': row[3], 'phone': row[4], 'address': row[5]}
    else:
        user = {}
    return render_template_string(profile_template, user=user)

@app.route('/profile/<int:user_id>')
def profile_view(user_id):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("SELECT id, username, name, email, phone, address FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        user = {'id': row[0], 'username': row[1], 'name': row[2], 'email': row[3], 'phone': row[4], 'address': row[5]}
    else:
        user = None
    return render_template_string(view_profile_template, user=user)

if __name__ == '__main__':
    app.run(debug=True)