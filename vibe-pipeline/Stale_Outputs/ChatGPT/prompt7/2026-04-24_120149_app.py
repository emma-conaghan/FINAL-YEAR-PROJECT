from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

# In-memory store for users
users = {}
next_id = 1

base_template = """
<!doctype html>
<title>User Profile App</title>
<h1>{{ title }}</h1>
{% block content %}{% endblock %}
"""

create_form = """
{% extends "base" %}
{% block content %}
<form method="post">
  Name: <input type="text" name="name" required><br>
  Email: <input type="email" name="email" required><br>
  Phone: <input type="text" name="phone"><br>
  Address: <input type="text" name="address"><br>
  <input type="submit" value="Create Account">
</form>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
{% endblock %}
"""

update_form = """
{% extends "base" %}
{% block content %}
<form method="post">
  Name: <input type="text" name="name" value="{{ user['name'] }}" required><br>
  Email: <input type="email" name="email" value="{{ user['email'] }}" required><br>
  Phone: <input type="text" name="phone" value="{{ user['phone'] }}"><br>
  Address: <input type="text" name="address" value="{{ user['address'] }}"><br>
  <input type="submit" value="Update Profile">
</form>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
{% endblock %}
"""

profile_page = """
{% extends "base" %}
{% block content %}
<p><strong>ID:</strong> {{ user_id }}</p>
<p><strong>Name:</strong> {{ user['name'] }}</p>
<p><strong>Email:</strong> {{ user['email'] }}</p>
<p><strong>Phone:</strong> {{ user['phone'] }}</p>
<p><strong>Address:</strong> {{ user['address'] }}</p>
<p><a href="{{ url_for('update', user_id=user_id) }}">Edit Profile</a></p>
<p><a href="{{ url_for('create') }}">Create New Account</a></p>
{% endblock %}
"""

@app.route("/")
def home():
    return redirect(url_for("create"))

@app.route("/create", methods=["GET", "POST"])
def create():
    global next_id
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()

        if not name or not email:
            error = "Name and Email are required."
        else:
            user_id = next_id
            next_id += 1
            users[user_id] = {
                "name": name,
                "email": email,
                "phone": phone,
                "address": address,
            }
            return redirect(url_for("profile", user_id=user_id))
    return render_template_string(create_form, error=error, title="Create Account")

@app.route("/update/<int:user_id>", methods=["GET", "POST"])
def update(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()

        if not name or not email:
            error = "Name and Email are required."
        else:
            user.update({
                "name": name,
                "email": email,
                "phone": phone,
                "address": address,
            })
            return redirect(url_for("profile", user_id=user_id))
    return render_template_string(update_form, user=user, error=error, title="Update Profile")

@app.route("/profile/<int:user_id>")
def profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(profile_page, user=user, user_id=user_id, title="Profile Details")

@app.context_processor
def inject_base():
    return dict(base=base_template)

if __name__ == "__main__":
    app.run(debug=True)