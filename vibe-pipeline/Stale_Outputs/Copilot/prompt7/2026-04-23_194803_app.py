from flask import Flask, render_template_string, request, redirect, url_for
import sqlite3

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        phone TEXT,
        address TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

signup_template = '''
<!DOCTYPE html>
<html>
<head><title>Sign Up</title></head>
<body>
<h2>Create Account</h2>
<form method="post">
    Name: <input type="text" name="name" required><br>
    Email: <input type="email" name="email" required><br>
    Phone: <input type="text" name="phone"><br>
    Address: <input type="text" name="address"><br>
    <input type="submit" value="Sign Up">
</form>
{% if error %}<p style="color:red;">{{error}}</p>{% endif %}
</body>
</html>
'''

update_template = '''
<!DOCTYPE html>
<html>
<head><title>Update Profile</title></head>
<body>
<h2>Update Profile</h2>
<form method="post">
    Name: <input type="text" name="name" value="{{user['name']}}" required><br>
    Email: <input type="email" name="email" value="{{user['email']}}" required><br>
    Phone: <input type="text" name="phone" value="{{user['phone']}}"><br>
    Address: <input type="text" name="address" value="{{user['address']}}"><br>
    <input type="submit" value="Update">
</form>
{% if error %}<p style="color:red;">{{error}}</p>{% endif %}
</body>
</html>
'''

profile_template = '''
<!DOCTYPE html>
<html>
<head><title>User Profile</title></head>
<body>
<h2>User Profile</h2>
{% if user %}
<ul>
    <li>ID: {{user['id']}}</li>
    <li>Name: {{user['name']}}</li>
    <li>Email: {{user['email']}}</li>
    <li>Phone: {{user['phone']}}</li>
    <li>Address: {{user['address']}}</li>
</ul>
<a href="{{url_for('update', user_id=user['id'])}}">Update Profile</a>
{% else %}
<p>User not found.</p>
{% endif %}
</body>
</html>
'''

home_template = '''
<!DOCTYPE html>
<html>
<head><title>Home</title></head>
<body>
<h2>Welcome</h2>
<a href="{{url_for('signup')}}">Create Account</a><br>
<form method="get" action="/profile">
    <label>View Profile by ID:</label>
    <input type="number" name="user_id" required>
    <input type="submit" value="View">
</form>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(home_template)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        phone = request.form.get('phone', '')
        address = request.form.get('address', '')
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (name, email, phone, address) VALUES (?, ?, ?, ?)",
                      (name, email, phone, address))
            conn.commit()
            user_id = c.lastrowid
            conn.close()
            return redirect(url_for('profile', user_id=user_id))
        except sqlite3.IntegrityError:
            error = "Email already in use."
    return render_template_string(signup_template, error=error)

def get_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, name, email, phone, address FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'name': row[1], 'email': row[2], 'phone': row[3], 'address': row[4]}
    else:
        return None

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update(user_id):
    error = None
    user = get_user(user_id)
    if not user:
        return render_template_string(profile_template, user=None)
    if request.method == 'POST':
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        phone = request.form.get('phone', '')
        address = request.form.get('address', '')
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("UPDATE users SET name=?, email=?, phone=?, address=? WHERE id=?",
                      (name, email, phone, address, user_id))
            conn.commit()
            conn.close()
            return redirect(url_for('profile', user_id=user_id))
        except sqlite3.IntegrityError:
            error = "Email already in use."
    return render_template_string(update_template, user=user, error=error)

@app.route('/profile')
def profile():
    user_id = request.args.get('user_id', '')
    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        user_id_int = None
    user = get_user(user_id_int) if user_id_int else None
    return render_template_string(profile_template, user=user)

if __name__ == '__main__':
    app.run(debug=True)