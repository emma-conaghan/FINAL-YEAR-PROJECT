from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_user_id = 1

CREATE_HTML = """
<!doctype html>
<title>Create Account</title>
<h1>Create Account</h1>
<form method=post>
  Name: <input type=text name=name required><br>
  Email: <input type=email name=email required><br>
  Phone Number: <input type=text name=phone><br>
  Address: <input type=text name=address><br>
  <input type=submit value=Create>
</form>
"""

UPDATE_HTML = """
<!doctype html>
<title>Update Profile</title>
<h1>Update Profile for User {{ user_id }}</h1>
<form method=post>
  Name: <input type=text name=name value="{{ user['name'] }}" required><br>
  Email: <input type=email name=email value="{{ user['email'] }}" required><br>
  Phone Number: <input type=text name=phone value="{{ user['phone'] }}"><br>
  Address: <input type=text name=address value="{{ user['address'] }}"><br>
  <input type=submit value=Update>
</form>
"""

VIEW_HTML = """
<!doctype html>
<title>View Profile</title>
<h1>Profile Details for User {{ user_id }}</h1>
<ul>
  <li>Name: {{ user['name'] }}</li>
  <li>Email: {{ user['email'] }}</li>
  <li>Phone Number: {{ user['phone'] }}</li>
  <li>Address: {{ user['address'] }}</li>
</ul>
<a href="{{ url_for('update', user_id=user_id) }}">Edit Profile</a>
"""

@app.route("/create", methods=["GET", "POST"])
def create():
    global next_user_id
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            return "Name and email are required.", 400
        user_id = next_user_id
        next_user_id += 1
        users[user_id] = {
            "name": name,
            "email": email,
            "phone": phone,
            "address": address
        }
        return redirect(url_for("view", user_id=user_id))
    return render_template_string(CREATE_HTML)

@app.route("/update/<int:user_id>", methods=["GET", "POST"])
def update(user_id):
    user = users.get(user_id)
    if user is None:
        abort(404)
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        address = request.form.get("address", "").strip()
        if not name or not email:
            return "Name and email are required.", 400
        users[user_id] = {
            "name": name,
            "email": email,
            "phone": phone,
            "address": address
        }
        return redirect(url_for("view", user_id=user_id))
    return render_template_string(UPDATE_HTML, user=user, user_id=user_id)

@app.route("/view/<int:user_id>")
def view(user_id):
    user = users.get(user_id)
    if user is None:
        abort(404)
    return render_template_string(VIEW_HTML, user=user, user_id=user_id)

if __name__ == "__main__":
    app.run(debug=True)