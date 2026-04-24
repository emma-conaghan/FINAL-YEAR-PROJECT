import os
from flask import Flask, render_template_string, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'internal_portal_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Internal Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        .flash { color: red; margin-bottom: 10px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> |
        {% if current_user.is_authenticated %}
            <a href="{{ url_for('dashboard') }}">Dashboard</a> |
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
          <div class="flash">{{ message }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h1>Welcome to the Internal Portal</h1>
    <p>Please log in to access the secure company areas.</p>
    {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
            
        is_first_user = User.query.count() == 0
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_pw, is_admin=is_first_user)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
        
    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h2>Register</h2>
    <form method="POST">
        Username: <input type="text" name="username" required><br><br>
        Password: <input type="password" name="password" required><br><br>
        <input type="submit" value="Sign Up">
    </form>
    {% endblock %}
    """)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password.')
        
    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h2>Login</h2>
    <form method="POST">
        Username: <input type="text" name="username" required><br><br>
        Password: <input type="password" name="password" required><br><br>
        <input type="submit" value="Login">
    </form>
    {% endblock %}
    """)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h2>User Dashboard</h2>
    <p>Hello, {{ current_user.username }}! Welcome to your private portal page.</p>
    <p>Internal News: The coffee machine is now fixed on the 3rd floor.</p>
    {% endblock %}
    """)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied: Admins only.')
        return redirect(url_for('dashboard'))
    
    users = User.query.all()
    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h2>Admin Area: Registered Users</h2>
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
    """, users=users)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)