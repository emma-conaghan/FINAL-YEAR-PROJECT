from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

signup_form = """
<h2>Create Account</h2>
<form method="post">
  Name: <input name="name"><br>
  Email: <input name="email"><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <input type="submit" value="Sign Up">
</form>
"""

update_form = """
<h2>Update Profile for User {{ user_id }}</h2>
<form method="post">
  Name: <input name="name" value="{{ user['name'] }}"><br>
  Email: <input name="email" value="{{ user['email'] }}"><br>
  Phone: <input name="phone" value="{{ user['phone'] }}"><br>
  Address: <input name="address" value="{{ user['address'] }}"><br>
  <input type="submit" value="Update">
</form>
"""

profile_view = """
<h2>Profile Details for User {{ user_id }}</h2>
<ul>
  <li>Name: {{ user['name'] }}</li>
  <li>Email: {{ user['email'] }}</li>
  <li>Phone: {{ user['phone'] }}</li>
  <li>Address: {{ user['address'] }}</li>
</ul>
<a href="{{ url_for('update_profile', user_id=user_id) }}">Edit Profile</a>
"""

@app.route("/", methods=["GET", "POST"])
def signup():
    global next_id
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if name and email:
            user_id = next_id
            next_id += 1
            users[user_id] = {"name": name, "email": email, "phone": phone, "address": address}
            return redirect(url_for("view_profile", user_id=user_id))
        return render_template_string(signup_form + "<p style='color:red'>Name and Email are required.</p>")
    return render_template_string(signup_form)

@app.route("/user/<int:user_id>", methods=["GET"])
def view_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(profile_view, user_id=user_id, user=user)

@app.route("/user/<int:user_id>/edit", methods=["GET", "POST"])
def update_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if name and email:
            user.update({"name": name, "email": email, "phone": phone, "address": address})
            return redirect(url_for("view_profile", user_id=user_id))
        return render_template_string(update_form + "<p style='color:red'>Name and Email are required.</p>", user_id=user_id, user=user)
    return render_template_string(update_form, user_id=user_id, user=user)

if __name__ == "__main__":
    app.run(debug=True)