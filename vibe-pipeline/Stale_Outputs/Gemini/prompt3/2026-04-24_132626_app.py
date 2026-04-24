import os
from flask import Flask, render_template_string, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-999'
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
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        .container { max-width: 600px; margin: auto; border: 1px solid #ccc; padding: 20px; border-radius: 8px; }
        .nav { margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .nav a { margin-right: 15px; text-decoration: none; color: blue; }
        input { display: block; width: 100%; margin-bottom: 10px; padding: 8px; box-sizing: border-box; }
        button { background: #28a745; color: white; border: none; padding: 10px 15px; cursor: pointer; border-radius: 4px; }
        .error { color: red; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav">
            {% if current_user.is_authenticated %}
                <a href="{{ url_for('home') }}">Home</a>
                {% if current_user.is_admin %}
                    <a href="{{ url_for('admin') }}">Admin Panel</a>
                {% endif %}
                <a href="{{ url_for('logout') }}">Logout ({{ current_user.username }})</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            {% for message in messages %}
              <p class="error">{{ message }}</p>
            {% endfor %}
          {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </div>
</body>
</html>
"""

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already exists.')
            return redirect(url_for('register'))
        
        # First user to register becomes admin
        is_admin = False
        if User.query.count() == 0:
            is_admin = True
            
        new_user = User(
            username=username, 
            password=generate_password_hash(password, method='sha256'),
            is_admin=is_admin
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
        
    return render_template_string(BASE_HTML + """
    {% block content %}
    <h2>Register</h2>
    <form method="POST">
        <input name="username" placeholder="Username" required>
        <input name="password" type="password" placeholder="Password" required>
        <button type="submit">Sign Up</button>
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
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your credentials.')
            
    return render_template_string(BASE_HTML + """
    {% block content %}
    <h2>Login</h2>
    <form method="POST">
        <input name="username" placeholder="Username" required>
        <input name="password" type="password" placeholder="Password" required>
        <button type="submit">Login</button>
    </form>
    {% endblock %}
    """)

@app.route('/')
@login_required
def home():
    return render_template_string(BASE_HTML + """
    {% block content %}
    <h2>Welcome, {{ current_user.username }}!</h2>
    <p>This is the internal company portal dashboard.</p>
    {% if current_user.is_admin %}
    <p><strong>You have administrative privileges.</strong></p>
    {% endif %}
    {% endblock %}
    """)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash("Unauthorized access.")
        return redirect(url_for('home'))
    
    users = User.query.all()
    return render_template_string(BASE_HTML + """
    {% block content %}
    <h2>Admin Panel - User Management</h2>
    <table>
        <tr>
            <th>ID</th>
            <th>Username</th>
            <th>Admin Status</th>
        </tr>
        {% for user in users %}
        <tr>
            <td>{{ user.id }}</td>
            <td>{{ user.username }}</td>
            <td>{{ 'Yes' if user.is_admin else 'No' }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endblock %}
    """, users=users)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)