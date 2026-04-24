from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}  # user_id -> user data
next_id = 1

base_template = """
<!doctype html>
<title>{{ title }}</title>
<h1>{{ title }}</h1>
{% block content %}{% endblock %}
"""

create_account_template = """
{% extends "base" %}
{% block content %}
<form method="post">
  <label>Name: <input type="text" name="name" required></label><br>
  <label>Email: <input type="email" name="email" required></label><br>
  <label>Phone: <input type="text" name="phone"></label><br>
  <label>Address: <input type="text" name="address"></label><br>
  <input type="submit" value="Create Account">
</form>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
{% endblock %}
"""

edit_profile_template = """
{% extends "base" %}
{% block content %}
<form method="post">
  <label>Name: <input type="text" name="name" value="{{ user.name }}" required></label><br>
  <label>Email: <input type="email" name="email" value="{{ user.email }}" required></label><br>
  <label>Phone: <input type="text" name="phone" value="{{ user.phone }}"></label><br>
  <label>Address: <input type="text" name="address" value="{{ user.address }}"></label><br>
  <input type="submit" value="Update Profile">
</form>
{% if message %}<p style="color:green;">{{ message }}</p>{% endif %}
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
{% endblock %}
"""

view_profile_template = """
{% extends "base" %}
{% block content %}
{% if user %}
<ul>
  <li><b>ID:</b> {{ user_id }}</li>
  <li><b>Name:</b> {{ user.name }}</li>
  <li><b>Email:</b> {{ user.email }}</li>
  <li><b>Phone:</b> {{ user.phone }}</li>
  <li><b>Address:</b> {{ user.address }}</li>
</ul>
<a href="{{ url_for('edit_profile', user_id=user_id) }}">Edit Profile</a>
{% else %}
<p>User not found</p>
{% endif %}
{% endblock %}
"""

from collections import namedtuple
User = namedtuple("User", "name email phone address")

@app.route("/")
def home():
    return redirect(url_for("create_account"))

@app.route("/create", methods=["GET", "POST"])
def create_account():
    global next_id
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            error = "Name and email are required."
        else:
            user_id = next_id
            next_id += 1
            users[user_id] = User(name, email, phone, address)
            return redirect(url_for("view_profile", user_id=user_id))
    return render_template_string(
        create_account_template, title="Create Account", error=error
    , **{"base": base_template})

@app.route("/profile/<int:user_id>")
def view_profile(user_id):
    user = users.get(user_id)
    return render_template_string(
        view_profile_template, title="View Profile", user=user, user_id=user_id
    , **{"base": base_template})

@app.route("/profile/<int:user_id>/edit", methods=["GET", "POST"])
def edit_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    error = None
    message = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            error = "Name and email are required."
        else:
            users[user_id] = User(name, email, phone, address)
            message = "Profile updated successfully."
            user = users[user_id]
    return render_template_string(
        edit_profile_template, title="Edit Profile", user=user, message=message, error=error
    , **{"base": base_template})

if __name__ == "__main__":
    app.run(debug=True)