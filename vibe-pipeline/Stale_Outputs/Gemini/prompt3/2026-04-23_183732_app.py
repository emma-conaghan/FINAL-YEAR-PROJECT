from flask import Flask, render_template_string, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'company-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Internal Portal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f4; }
        .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 500px; margin: auto; }
        input { display: block; width: 100%; margin: 10px 0; padding: 10px; box-sizing: border-box; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; width: 100%; }
        nav { margin-bottom: 20px; text-align: center; }
        .error { color: red; }
    </style>
</head>
<body>
    <div class="container">
        <nav>
            <a href="{{ url_for('index') }}">Home</a>
            {% if current_user.is_authenticated %}
                | <a href="{{ url_for('welcome') }}">Dashboard</a>
                {% if current_user.is_admin %} | <a href="{{ url_for('admin') }}">Admin</a> {% endif %}
                | <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                | <a href="{{ url_for('login') }}">Login</a> | <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </nav>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            {% for message in messages %}<p class="error">{{ message }}</p>{% endfor %}
          {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

LOGIN_HTML = """
{% extends "base" %}
{% block content %}
<h2>Login</h2>
<form method="POST">
    <input name="username" placeholder="Username" required>
    <input name="password" type="password" placeholder="Password" required>
    <button type="submit">Login</button>
</form>
{% endblock %}
"""

REGISTER_HTML = """
{% extends "base" %}
{% block content %}
<h2>Register</h2>
<form method="POST">
    <input name="username" placeholder="Username" required>
    <input name="password" type="password" placeholder="Password" required>
    <button type="submit">Sign Up</button>
</form>
{% endblock %}
"""

WELCOME_HTML = """
{% extends "base" %}
{% block content %}
<h2>Welcome, {{ current_user.username }}!</h2>
<p>This is your internal company dashboard.</p>
{% if current_user.is_admin %}
<p><strong>You have administrative privileges.</strong></p>
{% endif %}
{% endblock %}
"""

ADMIN_HTML = """
{% extends "base" %}
{% block content %}
<h2>Admin Panel - User List</h2>
<ul>
    {% for user in users %}
        <li>{{ user.username }} (ID: {{ user.id }}) {% if user.is_admin %}[Admin]{% endif %}</li>
    {% endfor %}
</ul>
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(BASE_HTML + "{% block content %}<h2>Internal Portal</h2><p>Please login or register to continue.</p>{% endblock %}")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        is_admin = (User.query.count() == 0)
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_pw, is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!')
        return redirect(url_for('login'))
    return render_template_string(REGISTER_HTML, base=BASE_HTML)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('welcome'))
        flash('Invalid credentials')
    return render_template_string(LOGIN_HTML, base=BASE_HTML)

@app.route('/welcome')
@login_required
def welcome():
    return render_template_string(WELCOME_HTML, base=BASE_HTML)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied: Admins only')
        return redirect(url_for('welcome'))
    users = User.query.all()
    return render_template_string(ADMIN_HTML, base=BASE_HTML, users=users)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)