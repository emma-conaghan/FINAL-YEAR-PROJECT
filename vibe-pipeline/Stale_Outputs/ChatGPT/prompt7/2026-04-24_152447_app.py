from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

home_html = """
<h1>Welcome</h1>
<p><a href="{{ url_for('register') }}">Register</a></p>
<p>View profile by account ID: 
<form action="{{ url_for('view_profile') }}" method="get" style="display:inline;">
  <input name="id" type="number" min="1" required>
  <input type="submit" value="View">
</form>
</p>
"""

register_html = """
<h1>Register</h1>
<form method="post">
  Name: <input name="name" required><br>
  Email: <input name="email" type="email" required><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <input type="submit" value="Create Account">
</form>
<p><a href="{{ url_for('home') }}">Home</a></p>
"""

update_html = """
<h1>Update Profile for Account {{ user_id }}</h1>
<form method="post">
  Name: <input name="name" value="{{ name }}" required><br>
  Email: <input name="email" type="email" value="{{ email }}" required><br>
  Phone: <input name="phone" value="{{ phone }}"><br>
  Address: <input name="address" value="{{ address }}"><br>
  <input type="submit" value="Update Profile">
</form>
<p><a href="{{ url_for('view_profile_by_id', user_id=user_id) }}">Back to Profile</a></p>
<p><a href="{{ url_for('home') }}">Home</a></p>
"""

profile_html = """
<h1>Profile for Account {{ user_id }}</h1>
<ul>
  <li>Name: {{ name }}</li>
  <li>Email: {{ email }}</li>
  <li>Phone: {{ phone }}</li>
  <li>Address: {{ address }}</li>
</ul>
<p><a href="{{ url_for('update_profile', user_id=user_id) }}">Update Profile</a></p>
<p><a href="{{ url_for('home') }}">Home</a></p>
"""

@app.route("/")
def home():
    return render_template_string(home_html)

@app.route("/register", methods=["GET", "POST"])
def register():
    global next_id
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            return "Name and Email are required", 400
        user_id = next_id
        next_id += 1
        users[user_id] = {"name": name, "email": email, "phone": phone, "address": address}
        return redirect(url_for("view_profile_by_id", user_id=user_id))
    return render_template_string(register_html)

@app.route("/profile/<int:user_id>")
def view_profile_by_id(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(profile_html, user_id=user_id, **user)

@app.route("/profile", methods=["GET"])
def view_profile():
    try:
        user_id = int(request.args.get("id"))
    except (TypeError, ValueError):
        return "Invalid account ID", 400
    user = users.get(user_id)
    if not user:
        return "User not found", 404
    return redirect(url_for("view_profile_by_id", user_id=user_id))

@app.route("/profile/<int:user_id>/update", methods=["GET", "POST"])
def update_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            return "Name and Email are required", 400
        user.update({"name": name, "email": email, "phone": phone, "address": address})
        return redirect(url_for("view_profile_by_id", user_id=user_id))
    return render_template_string(update_html, user_id=user_id, **user)

if __name__ == "__main__":
    app.run(debug=True)