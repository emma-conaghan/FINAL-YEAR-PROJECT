from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
# users structure: {user_id: {"name":..., "email":..., "phone":..., "address":...}}

next_user_id = 1

create_account_html = """
<h2>Create Account</h2>
<form method="POST">
  Name:<br><input type="text" name="name" required><br>
  Email:<br><input type="email" name="email" required><br>
  Phone Number:<br><input type="text" name="phone" required><br>
  Address:<br><textarea name="address" required></textarea><br>
  <input type="submit" value="Create Account">
</form>
"""

update_profile_html = """
<h2>Update Profile for User ID: {{user_id}}</h2>
<form method="POST">
  Name:<br><input type="text" name="name" value="{{profile.name}}" required><br>
  Email:<br><input type="email" name="email" value="{{profile.email}}" required><br>
  Phone Number:<br><input type="text" name="phone" value="{{profile.phone}}" required><br>
  Address:<br><textarea name="address" required>{{profile.address}}</textarea><br>
  <input type="submit" value="Update Profile">
</form>
"""

view_profile_html = """
<h2>Profile Details for User ID: {{user_id}}</h2>
<ul>
  <li><b>Name:</b> {{profile.name}}</li>
  <li><b>Email:</b> {{profile.email}}</li>
  <li><b>Phone Number:</b> {{profile.phone}}</li>
  <li><b>Address:</b> {{profile.address}}</li>
</ul>
<a href="{{url_for('update_profile', user_id=user_id)}}">Edit Profile</a><br>
<a href="{{url_for('create_account')}}">Create New Account</a>
"""

@app.route("/")
def index():
    return redirect(url_for("create_account"))

@app.route("/create_account", methods=["GET", "POST"])
def create_account():
    global next_user_id
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if name and email and phone and address:
            user_id = next_user_id
            next_user_id += 1
            users[user_id] = {"name": name, "email": email, "phone": phone, "address": address}
            return redirect(url_for("view_profile", user_id=user_id))
        else:
            return render_template_string(create_account_html + "<p style='color:red;'>All fields are required.</p>")
    return render_template_string(create_account_html)

@app.route("/profile/<int:user_id>")
def view_profile(user_id):
    profile = users.get(user_id)
    if not profile:
        abort(404)
    return render_template_string(view_profile_html, profile=profile, user_id=user_id)

@app.route("/update_profile/<int:user_id>", methods=["GET", "POST"])
def update_profile(user_id):
    profile = users.get(user_id)
    if not profile:
        abort(404)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if name and email and phone and address:
            profile["name"] = name
            profile["email"] = email
            profile["phone"] = phone
            profile["address"] = address
            return redirect(url_for("view_profile", user_id=user_id))
        else:
            error = "<p style='color:red;'>All fields are required.</p>"
            return render_template_string(update_profile_html + error, profile=profile, user_id=user_id)
    return render_template_string(update_profile_html, profile=profile, user_id=user_id)

if __name__ == "__main__":
    app.run(debug=True)