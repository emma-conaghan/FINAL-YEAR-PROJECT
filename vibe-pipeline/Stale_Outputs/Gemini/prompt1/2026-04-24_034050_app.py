import os
from flask import Flask, render_template_string, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'internal-portal-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

LAYOUT = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        .error { color: red; }
        .success { color: green; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> |
        {% if current_user.is_authenticated %}
            <a href="{{ url_for('welcome') }}">Dashboard</a> |
            {% if current_user.is_admin %}
                <a href="{{ url_for('admin') }}">Admin Panel</a> |
            {% endif %}
            <a href="{{ url_for('logout') }}">Logout ({{ current_user.username }})</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> |
            <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for message in messages %}
          <p class="error">{{ message }}</p>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
"""

INDEX_HTML = """
{% extends "layout" %}
{% block content %}
<h1>Welcome to the Internal Company Portal</h1>
<p>Please log in or register to access your account.</p>
{% endblock %}
"""

LOGIN_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Login</h2>
<form method="POST">
    Username: <input type="text" name="username" required><br><br>
    Password: <input type="password" name="password" required><br><br>
    <input type="submit" value="Login">
</form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Register</h2>
<form method="POST">
    Username: <input type="text" name="username" required><br><br>
    Password: <input type="password" name="password" required><br><br>
    Admin Access: <input type="checkbox" name="is_admin"><br><br>
    <input type="submit" value="Register">
</form>
{% endblock %}
"""

WELCOME_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Welcome, {{ current_user.username }}!</h2>
<p>This is your private dashboard. You are successfully authenticated.</p>
{% endblock %}
"""

ADMIN_HTML = """
{% extends "layout" %}
{% block content %}
<h2>Admin Panel - User Management</h2>
<table>
    <tr>
        <th>ID</th>
        <th>Username</th>
        <th>Is Admin</th>
    </tr>
    {% for user in users %}
    <tr>
        <td>{{ user.id }}</td>
        <td>{{ user.username }}</td>
        <td>{{ user.is_admin }}</td>
    </tr>
    {% endfor %}
</table>
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(LAYOUT + INDEX_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        is_admin = True if request.form.get('is_admin') else False
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
        
        hashed_pw = generate_password_hash(password, method='sha256')
        new_user = User(username=username, password=hashed_pw, is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template_string(LAYOUT + REGISTER_HTML)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('welcome'))
        else:
            flash('Invalid username or password.')
    return render_template_string(LAYOUT + LOGIN_HTML)

@app.route('/welcome')
@login_required
def welcome():
    return render_template_string(LAYOUT + WELCOME_HTML)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied: Admins only.')
        return redirect(url_for('welcome'))
    users = User.query.all()
    return render_template_string(LAYOUT + ADMIN_HTML, users=users)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)