from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

users = {}  # account_id: {name, email, phone, address}
next_id = 1

home_html = """
<h2>Welcome</h2>
<a href="/register">Register Account</a>
<br>
<a href="/update">Update Profile</a>
<br>
<a href="/view">View Profile by Account ID</a>
"""

register_html = """
<h2>Register Account</h2>
<form method="post">
  Name: <input name="name"><br>
  Email: <input name="email"><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <input type="submit" value="Register">
</form>
"""

update_html = """
<h2>Update Profile</h2>
<form method="post">
  Account ID: <input name="account_id"><br>
  Name: <input name="name"><br>
  Email: <input name="email"><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <input type="submit" value="Update">
</form>
"""

view_html = """
<h2>View Profile</h2>
<form method="get">
  Account ID: <input name="account_id">
  <input type="submit" value="View">
</form>
{% if user %}
<hr>
<b>Account ID:</b> {{ account_id }}<br>
<b>Name:</b> {{ user['name'] }}<br>
<b>Email:</b> {{ user['email'] }}<br>
<b>Phone:</b> {{ user['phone'] }}<br>
<b>Address:</b> {{ user['address'] }}<br>
{% elif account_id %}
Not found.
{% endif %}
"""

@app.route("/")
def home():
    return render_template_string(home_html)

@app.route("/register", methods=["GET", "POST"])
def register():
    global users, next_id
    if request.method == "POST":
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        phone = request.form.get("phone", "")
        address = request.form.get("address", "")
        account_id = str(next_id)
        users[account_id] = {"name": name, "email": email, "phone": phone, "address": address}
        next_id += 1
        return f"Registered. Your Account ID is {account_id}. <a href='/'>Home</a>"
    return render_template_string(register_html)

@app.route("/update", methods=["GET", "POST"])
def update():
    global users
    msg = ""
    if request.method == "POST":
        account_id = request.form.get("account_id", "")
        if account_id in users:
            users[account_id]["name"] = request.form.get("name", "")
            users[account_id]["email"] = request.form.get("email", "")
            users[account_id]["phone"] = request.form.get("phone", "")
            users[account_id]["address"] = request.form.get("address", "")
            msg = "Updated."
        else:
            msg = "Account ID not found."
        msg += " <a href='/'>Home</a>"
        return msg
    return render_template_string(update_html)

@app.route("/view", methods=["GET"])
def view():
    account_id = request.args.get("account_id")
    user = users.get(account_id) if account_id else None
    return render_template_string(view_html, user=user, account_id=account_id)

if __name__ == "__main__":
    app.run(debug=True)