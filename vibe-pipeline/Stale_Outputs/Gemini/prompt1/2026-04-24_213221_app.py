from flask import Flask, request, redirect, url_for, render_template_string
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key'
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
<head><title>Company Portal</title></head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a> |
        {% if current_user.is_authenticated %}
            <a href="{{ url_for('logout') }}">Logout</a>
            {% if current_user.is_admin %} | <a href="{{ url_for('admin') }}">Admin Area</a>{% endif %}
        {% else %}
            <a href="{{ url_for('login') }}">Login</a> | <a href="{{ url_for('register') }}">Register</a>
        {% endif %}
    </nav>
    <hr>
    {% with messages = get_flashed_messages() %}{% if messages %}<ul>{% for m in messages %}<li>{{ m }}</li>{% endfor %}</ul>{% endif %}{% endwith %}
    {% block content %}{% endblock %}
</body>
</html>
"""

@app.route('/')
def index():
    content = "<h1>Welcome to the Internal Portal</h1>"
    if current_user.is_authenticated:
        content += f"<p>Hello, {current_user.username}!</p>"
    else:
        content += "<p>Please login or register to continue.</p>"
    return render_template_string(BASE_HTML.replace('{% block content %}{% endblock %}', content))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            return "Username exists"
        is_admin = True if User.query.count() == 0 else False
        new_user = User(username=username, password=generate_password_hash(password), is_admin=is_admin)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    
    form = """
    <h2>Register</h2>
    <form method="POST">
        Username: <input type="text" name="username"><br>
        Password: <input type="password" name="password"><br>
        <input type="submit" value="Sign Up">
    </form>
    """
    return render_template_string(BASE_HTML.replace('{% block content %}{% endblock %}', form))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('index'))
        return "Invalid credentials"
    
    form = """
    <h2>Login</h2>
    <form method="POST">
        Username: <input type="text" name="username"><br>
        Password: <input type="password" name="password"><br>
        <input type="submit" value="Login">
    </form>
    """
    return render_template_string(BASE_HTML.replace('{% block content %}{% endblock %}', form))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        return "Access Denied", 403
    users = User.query.all()
    user_list = "<h2>Registered Users</h2><ul>"
    for u in users:
        user_list += f"<li>{u.username} (Admin: {u.is_admin})</li>"
    user_list += "</ul>"
    return render_template_string(BASE_HTML.replace('{% block content %}{% endblock %}', user_list))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)