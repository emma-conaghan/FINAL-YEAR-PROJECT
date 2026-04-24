import os
from flask import Flask, render_template_string, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'company-portal-secret-key-12345'
app.config['SQLALCHEMY_DATABASE_DATABASE_URI'] = 'sqlite:///portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

BASE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        nav a { margin-right: 15px; text-decoration: none; color: #007bff; }
        .error { color: red; }
        .success { color: green; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    </style>
</head>
<body>
    <nav>
        {% if session.get('user_id') %}
            <a href="{{ url_for('index') }}">Home</a>
            {% if session.get('is_admin') %}
                <a href="{{ url_for('admin') }}">Admin Panel</a>
            {% endif %}
            <a href="{{ url_for('logout') }}">Logout ({{ session.get('username') }})</a>
        {% else %}
            <a href="{{ url_for('login') }}">Login</a>
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

@app.route('/')
@login_required
def index():
    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h1>Welcome to the Internal Portal, {{ session['username'] }}!</h1>
    <p>This is a secure area for company employees.</p>
    {% endblock %}
    """)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        is_admin = True if User.query.count() == 0 else False
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw, is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))

    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h2>Register</h2>
    <form method="POST">
        <div><label>Username:</label><br><input type="text" name="username" required></div>
        <div><label>Password:</label><br><input type="password" name="password" required></div>
        <br><button type="submit">Sign Up</button>
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
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            return redirect(url_for('index'))
        
        flash('Invalid username or password')
    
    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h2>Login</h2>
    <form method="POST">
        <div><label>Username:</label><br><input type="text" name="username" required></div>
        <div><label>Password:</label><br><input type="password" name="password" required></div>
        <br><button type="submit">Login</button>
    </form>
    {% endblock %}
    """)

@app.route('/admin')
@login_required
def admin():
    if not session.get('is_admin'):
        flash('Access denied: Admin only.')
        return redirect(url_for('index'))
    
    users = User.query.all()
    return render_template_string(BASE_TEMPLATE + """
    {% block content %}
    <h2>Admin Panel - Registered Users</h2>
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
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)