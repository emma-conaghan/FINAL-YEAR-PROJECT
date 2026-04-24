from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

# In-memory storage for users
users = {}
next_user_id = 1

create_account_template = """
<!doctype html>
<title>Create Account</title>
<h1>Create Account</h1>
<form method="post">
  Name: <input type="text" name="name" required><br>
  Email: <input type="email" name="email" required><br>
  Phone: <input type="text" name="phone" required><br>
  Address: <input type="text" name="address" required><br>
  <input type="submit" value="Create">
</form>
"""

update_profile_template = """
<!doctype html>
<title>Update Profile</title>
<h1>Update Profile for User {{ user_id }}</h1>
<form method="post">
  Name: <input type="text" name="name" value="{{ user['name'] }}" required><br>
  Email: <input type="email" name="email" value="{{ user['email'] }}" required><br>
  Phone: <input type="text" name="phone" value="{{ user['phone'] }}" required><br>
  Address: <input type="text" name="address" value="{{ user['address'] }}" required><br>
  <input type="submit" value="Update">
</form>
"""

view_profile_template = """
<!doctype html>
<title>View Profile</title>
<h1>Profile Details for User {{ user_id }}</h1>
<ul>
  <li>Name: {{ user['name'] }}</li>
  <li>Email: {{ user['email'] }}</li>
  <li>Phone: {{ user['phone'] }}</li>
  <li>Address: {{ user['address'] }}</li>
</ul>
<a href="{{ url_for('update_profile', user_id=user_id) }}">Update Profile</a><br>
<a href="{{ url_for('create_account') }}">Create New Account</a>
"""

@app.route('/')
def index():
    return redirect(url_for('create_account'))

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    global next_user_id
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not (name and email and phone and address):
            return render_template_string(create_account_template)  # Simple fallback, no error message

        user_id = next_user_id
        users[user_id] = {'name': name, 'email': email, 'phone': phone, 'address': address}
        next_user_id += 1
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(create_account_template)

@app.route('/profile/<int:user_id>', methods=['GET'])
def view_profile(user_id):
    user = users.get(user_id)
    if user is None:
        abort(404)
    return render_template_string(view_profile_template, user=user, user_id=user_id)

@app.route('/profile/<int:user_id>/edit', methods=['GET', 'POST'])
def update_profile(user_id):
    user = users.get(user_id)
    if user is None:
        abort(404)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()

        if not (name and email and phone and address):
            return render_template_string(update_profile_template, user=user, user_id=user_id)

        user['name'] = name
        user['email'] = email
        user['phone'] = phone
        user['address'] = address

        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(update_profile_template, user=user, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)