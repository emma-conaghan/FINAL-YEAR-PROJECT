import os
from flask import Flask, request, redirect, url_for, session, render_template_string, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        admin_user = User(
            username='admin', 
            password=generate_password_hash('admin123'), 
            is_admin=True
        )
        db.session.add(admin_user)
        db.session.commit()

BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Company Portal</title>
    <style>
        body { font-family: sans-serif; margin: 40px; line-height: 1.6; }
        nav { margin-bottom: 20px; border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        .error { color: red; }
    </style>
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> | 
        {% if 'user_id' in session %}
            <a href="{{ url_for('admin_panel') }}">Admin</a> | 
            <a href="{{ url_for('logout') }}">Logout</a>
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

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(id=session['user_id']).first()
    return render_template_string(BASE_HTML + """
        {% block content %}
            <h1>Welcome, {{ name }}!</h1>
            <p>You are successfully logged into the internal company portal.</p>
        {% endblock %}
    """, name=user.username)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template_string(BASE_HTML + """
        {% block content %}
            <h2>Register</h2>
            <form method="post">
                <label>Username:</label><br>
                <input type="text" name="username" required><br>
                <label>Password:</label><br>
                <input type="password" name="password" required><br><br>
                <button type="submit">Create Account</button>
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
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template_string(BASE_HTML + """
        {% block content %}
            <h2>Login</h2>
            <form method="post">
                <label>Username:</label><br>
                <input type="text" name="username" required><br>
                <label>Password:</label><br>
                <input type="password" name="password" required><br><br>
                <button type="submit">Login</button>
            </form>
        {% endblock %}
    """)

@app.route('/admin')
def admin_panel():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    current_user = User.query.filter_by(id=session['user_id']).first()
    if not current_user or not current_user.is_admin:
        return "Access Denied: Admins Only", 403
    all_users = User.query.all()
    return render_template_string(BASE_HTML + """
        {% block content %}
            <h2>Admin Area - Registered Users</h2>
            <table border="1" cellpadding="5" style="border-collapse: collapse;">
                <tr>
                    <th>ID</th>
                    <th>Username</th>
                    <th>Is Admin</th>
                </tr>
                {% for u in users %}
                <tr>
                    <td>{{ u.id }}</td>
                    <td>{{ u.username }}</td>
                    <td>{{ u.is_admin }}</td>
                </tr>
                {% endfor %}
            </table>
        {% endblock %}
    """, users=all_users)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)