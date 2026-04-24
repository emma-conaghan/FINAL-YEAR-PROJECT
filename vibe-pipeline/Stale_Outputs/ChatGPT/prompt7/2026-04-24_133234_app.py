from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

# In-memory storage for users
# Structure: {user_id: {'name': ..., 'email': ..., 'phone': ..., 'address': ...}}
users = {}
next_user_id = 1

create_template = """
<!doctype html>
<title>Create Account</title>
<h1>Create Account</h1>
<form method="post">
  Name: <input type="text" name="name" required><br>
  Email: <input type="email" name="email" required><br>
  Phone: <input type="text" name="phone"><br>
  Address: <input type="text" name="address"><br>
  <input type="submit" value="Create Account">
</form>
{% if user_id %}
<p>Account created! Your user ID is <a href="{{ url_for('view_profile', user_id=user_id) }}">{{ user_id }}</a></p>
{% endif %}
"""

update_template = """
<!doctype html>
<title>Update Profile</title>
<h1>Update Profile (User ID: {{ user_id }})</h1>
<form method="post">
  Name: <input type="text" name="name" value="{{ user.name }}" required><br>
  Email: <input type="email" name="email" value="{{ user.email }}" required><br>
  Phone: <input type="text" name="phone" value="{{ user.phone }}"><br>
  Address: <input type="text" name="address" value="{{ user.address }}"><br>
  <input type="submit" value="Update Profile">
</form>
<p><a href="{{ url_for('view_profile', user_id=user_id) }}">View profile</a></p>
"""

view_template = """
<!doctype html>
<title>View Profile</title>
<h1>Profile (User ID: {{ user_id }})</h1>
<ul>
  <li><strong>Name:</strong> {{ user.name }}</li>
  <li><strong>Email:</strong> {{ user.email }}</li>
  <li><strong>Phone:</strong> {{ user.phone }}</li>
  <li><strong>Address:</strong> {{ user.address }}</li>
</ul>
<p><a href="{{ url_for('update_profile', user_id=user_id) }}">Edit Profile</a></p>
<p><a href="{{ url_for('create_account') }}">Create New Account</a></p>
"""

@app.route("/", methods=["GET", "POST"])
def create_account():
    global next_user_id
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            return render_template_string(create_template)
        user_id = next_user_id
        next_user_id += 1
        users[user_id] = {
            "name": name,
            "email": email,
            "phone": phone,
            "address": address,
        }
        return render_template_string(create_template, user_id=user_id)
    return render_template_string(create_template)

@app.route("/profile/<int:user_id>", methods=["GET"])
def view_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(view_template, user=user, user_id=user_id)

@app.route("/profile/<int:user_id>/edit", methods=["GET", "POST"])
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
            return render_template_string(update_template, user=user, user_id=user_id)
        user.update({
            "name": name,
            "email": email,
            "phone": phone,
            "address": address,
        })
        return redirect(url_for("view_profile", user_id=user_id))
    return render_template_string(update_template, user=user, user_id=user_id)

if __name__ == "__main__":
    app.run(debug=True)