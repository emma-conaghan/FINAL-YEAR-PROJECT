from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

form_create = '''
<h2>Create Account</h2>
<form method="post">
  Name: <input name="name" required><br>
  Email: <input name="email" type="email" required><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <button type="submit">Create</button>
</form>
'''

form_update = '''
<h2>Update Profile (User ID: {{id}})</h2>
<form method="post">
  Name: <input name="name" value="{{user['name']}}" required><br>
  Email: <input name="email" type="email" value="{{user['email']}}" required><br>
  Phone: <input name="phone" value="{{user['phone']}}"><br>
  Address: <input name="address" value="{{user['address']}}"><br>
  <button type="submit">Update</button>
</form>
'''

view_profile = '''
<h2>Profile Details (User ID: {{id}})</h2>
<ul>
  <li>Name: {{user['name']}}</li>
  <li>Email: {{user['email']}}</li>
  <li>Phone: {{user['phone']}}</li>
  <li>Address: {{user['address']}}</li>
</ul>
<a href="{{ url_for('update_profile', user_id=id) }}">Update Profile</a><br>
<a href="{{ url_for('create_account') }}">Create New Account</a>
'''

@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("create_account"))

@app.route("/create", methods=["GET", "POST"])
def create_account():
    global next_id
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            return render_template_string(form_create + "<p style='color:red;'>Name and Email required.</p>")
        user_id = next_id
        next_id += 1
        users[user_id] = {
            "name": name,
            "email": email,
            "phone": phone,
            "address": address
        }
        return redirect(url_for("view_profile_page", user_id=user_id))
    return render_template_string(form_create)

@app.route("/profile/<int:user_id>", methods=["GET"])
def view_profile_page(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(view_profile, id=user_id, user=user)

@app.route("/update/<int:user_id>", methods=["GET", "POST"])
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
            return render_template_string(form_update + "<p style='color:red;'>Name and Email required.</p>", id=user_id, user=user)
        user.update({
            "name": name,
            "email": email,
            "phone": phone,
            "address": address
        })
        return redirect(url_for("view_profile_page", user_id=user_id))
    return render_template_string(form_update, id=user_id, user=user)

if __name__ == "__main__":
    app.run(debug=True)