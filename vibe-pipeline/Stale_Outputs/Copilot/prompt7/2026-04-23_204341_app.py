from flask import Flask, render_template_string, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

users = {}
profiles = {}

register_template = '''
<!DOCTYPE html>
<html>
<head><title>Register</title></head>
<body>
    <h2>Register Account</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Register">
    </form>
    <p>Already have an account? <a href="/login">Login</a></p>
    {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
</body>
</html>
'''

login_template = '''
<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
    <h2>Login</h2>
    <form method="post">
        Username: <input type="text" name="username" required><br>
        Password: <input type="password" name="password" required><br>
        <input type="submit" value="Login">
    </form>
    <p>Don't have an account? <a href="/register">Register</a></p>
    {% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
</body>
</html>
'''

profile_template = '''
<!DOCTYPE html>
<html>
<head><title>Profile</title></head>
<body>
    <h2>Your Profile</h2>
    <form method="post">
        Name: <input type="text" name="name" value="{{ profile.name }}"><br>
        Email: <input type="email" name="email" value="{{ profile.email }}"><br>
        Phone: <input type="text" name="phone" value="{{ profile.phone }}"><br>
        Address: <input type="text" name="address" value="{{ profile.address }}"><br>
        <input type="submit" value="Update Profile">
    </form>
    <p><a href="/profile/{{ user_id }}">View Profile Page</a></p>
    <p><a href="/logout">Logout</a></p>
    {% if msg %}<p style="color:green;">{{ msg }}</p>{% endif %}
</body>
</html>
'''

profile_view_template = '''
<!DOCTYPE html>
<html>
<head><title>User Profile</title></head>
<body>
    <h2>User Profile (ID: {{ user_id }})</h2>
    {% if profile %}
        <ul>
            <li>Name: {{ profile.name }}</li>
            <li>Email: {{ profile.email }}</li>
            <li>Phone: {{ profile.phone }}</li>
            <li>Address: {{ profile.address }}</li>
        </ul>
    {% else %}
        <p>No profile found for this user.</p>
    {% endif %}
    <p><a href="/">Back to Home</a></p>
</body>
</html>
'''

home_template = '''
<!DOCTYPE html>
<html>
<head><title>Home</title></head>
<body>
    <h2>Welcome</h2>
    {% if user_id %}
        <p>You are logged in as {{ user_id }}.</p>
        <p><a href="/profile/edit">Edit Profile</a></p>
        <p><a href="/profile/{{ user_id }}">View Profile</a></p>
        <p><a href="/logout">Logout</a></p>
    {% else %}
        <p><a href="/register">Register</a> or <a href="/login">Login</a></p>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    user_id = session.get('user_id')
    return render_template_string(home_template, user_id=user_id)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        if not username or not password:
            error = "All fields are required."
        elif username in users:
            error = "Username already exists."
        else:
            users[username] = generate_password_hash(password)
            profiles[username] = {'name': '', 'email': '', 'phone': '', 'address': ''}
            session['user_id'] = username
            return redirect(url_for('home'))
    return render_template_string(register_template, error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['user_id'] = username
            return redirect(url_for('home'))
        else:
            error = "Invalid username or password."
    return render_template_string(login_template, error=error)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/profile/edit', methods=['GET', 'POST'])
def edit_profile():
    user_id = session.get('user_id')
    if not user_id or user_id not in profiles:
        return redirect(url_for('login'))
    msg = None
    profile = profiles[user_id]
    if request.method == 'POST':
        profile['name'] = request.form.get('name', '')
        profile['email'] = request.form.get('email', '')
        profile['phone'] = request.form.get('phone', '')
        profile['address'] = request.form.get('address', '')
        msg = "Profile updated."
    class ProfileObj:
        def __init__(self, d):
            self.name = d.get('name', '')
            self.email = d.get('email', '')
            self.phone = d.get('phone', '')
            self.address = d.get('address', '')
    return render_template_string(profile_template, profile=ProfileObj(profile), user_id=user_id, msg=msg)

@app.route('/profile/<user_id>')
def view_profile(user_id):
    profile = profiles.get(user_id)
    class ProfileObj:
        def __init__(self, d):
            self.name = d.get('name', '')
            self.email = d.get('email', '')
            self.phone = d.get('phone', '')
            self.address = d.get('address', '')
    profile_obj = ProfileObj(profile) if profile else None
    return render_template_string(profile_view_template, profile=profile_obj, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)