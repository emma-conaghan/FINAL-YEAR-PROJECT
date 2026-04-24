from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

INDEX_HTML = """
<h1>Simple User Account System</h1>
<a href="{{ url_for('create_account') }}">Create Account</a><br>
<a href="{{ url_for('view_profile_form') }}">View Profile by ID</a>
"""

CREATE_HTML = """
<h1>Create Account</h1>
<form method="post">
  Name: <input name="name" required><br>
  Email: <input name="email" type="email" required><br>
  Phone Number: <input name="phone"><br>
  Address: <input name="address"><br>
  <input type="submit" value="Create">
</form>
<a href="{{ url_for('index') }}">Home</a>
"""

UPDATE_HTML = """
<h1>Update Profile</h1>
<form method="post">
  Name: <input name="name" value="{{ user['name'] }}" required><br>
  Email: <input name="email" type="email" value="{{ user['email'] }}" required><br>
  Phone Number: <input name="phone" value="{{ user['phone'] }}"><br>
  Address: <input name="address" value="{{ user['address'] }}"><br>
  <input type="submit" value="Update">
</form>
<a href="{{ url_for('index') }}">Home</a>
"""

VIEW_HTML = """
<h1>View Profile</h1>
{% if user %}
  <p><b>ID:</b> {{ user_id }}</p>
  <p><b>Name:</b> {{ user['name'] }}</p>
  <p><b>Email:</b> {{ user['email'] }}</p>
  <p><b>Phone Number:</b> {{ user['phone'] }}</p>
  <p><b>Address:</b> {{ user['address'] }}</p>
  <a href="{{ url_for('update_profile', user_id=user_id) }}">Edit Profile</a><br>
{% else %}
  <p>User not found.</p>
{% endif %}
<a href="{{ url_for('index') }}">Home</a>
"""

VIEW_FORM_HTML = """
<h1>View Profile by Account ID</h1>
<form action="{{ url_for('view_profile') }}" method="get">
  Account ID: <input name="user_id" type="number" required><br>
  <input type="submit" value="View">
</form>
<a href="{{ url_for('index') }}">Home</a>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/create", methods=["GET", "POST"])
def create_account():
    global next_id
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            return "Name and Email required", 400
        user_id = next_id
        users[user_id] = {
            "name": name,
            "email": email,
            "phone": phone,
            "address": address,
        }
        next_id += 1
        return redirect(url_for("view_profile", user_id=user_id))
    return render_template_string(CREATE_HTML)

@app.route("/profile/<int:user_id>", methods=["GET"])
def view_profile(user_id=None):
    user = users.get(user_id)
    return render_template_string(VIEW_HTML, user=user, user_id=user_id)

@app.route("/profile/update/<int:user_id>", methods=["GET", "POST"])
def update_profile(user_id=None):
    user = users.get(user_id)
    if not user:
        abort(404)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            return "Name and Email required", 400
        user.update({
            "name": name,
            "email": email,
            "phone": phone,
            "address": address,
        })
        return redirect(url_for("view_profile", user_id=user_id))
    return render_template_string(UPDATE_HTML, user=user)

@app.route("/view")
def view_profile_form():
    return render_template_string(VIEW_FORM_HTML)

if __name__ == "__main__":
    app.run(debug=True)