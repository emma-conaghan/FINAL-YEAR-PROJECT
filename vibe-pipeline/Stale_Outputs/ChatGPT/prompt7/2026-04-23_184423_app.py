from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

layout = """
<!doctype html>
<title>User Profile App</title>
<h1><a href="{{ url_for('index') }}">User Profile App</a></h1>
{% block body %}{% endblock %}
"""

index_template = """
{% extends "layout" %}
{% block body %}
<p><a href="{{ url_for('create_account') }}">Create Account</a></p>
<p>View Profile by Account ID:</p>
<form method="get" action="{{ url_for('view_profile', account_id=0)[:-1] }}">
    <input type="number" name="account_id" min="1" required>
    <button type="submit">View Profile</button>
</form>
{% endblock %}
"""

create_account_template = """
{% extends "layout" %}
{% block body %}
<h2>Create Account</h2>
<form method="post">
  <label>Name: <input type="text" name="name" required></label><br><br>
  <label>Email: <input type="email" name="email" required></label><br><br>
  <label>Phone Number: <input type="tel" name="phone" required></label><br><br>
  <label>Address:<br><textarea name="address" rows="3" cols="30" required></textarea></label><br><br>
  <button type="submit">Create</button>
</form>
{% if error %}<p style="color:red;">{{ error }}</p>{% endif %}
{% endblock %}
"""

update_profile_template = """
{% extends "layout" %}
{% block body %}
<h2>Update Profile for Account ID {{ account_id }}</h2>
<form method="post">
  <label>Name: <input type="text" name="name" value="{{ user['name'] }}" required></label><br><br>
  <label>Email: <input type="email" name="email" value="{{ user['email'] }}" required></label><br><br>
  <label>Phone Number: <input type="tel" name="phone" value="{{ user['phone'] }}" required></label><br><br>
  <label>Address:<br><textarea name="address" rows="3" cols="30" required>{{ user['address'] }}</textarea></label><br><br>
  <button type="submit">Update</button>
</form>
{% if message %}<p style="color:green;">{{ message }}</p>{% endif %}
{% endblock %}
"""

view_profile_template = """
{% extends "layout" %}
{% block body %}
<h2>Profile for Account ID {{ account_id }}</h2>
<p><strong>Name:</strong> {{ user['name'] }}</p>
<p><strong>Email:</strong> {{ user['email'] }}</p>
<p><strong>Phone Number:</strong> {{ user['phone'] }}</p>
<p><strong>Address:</strong> {{ user['address'] }}</p>
<p><a href="{{ url_for('update_profile', account_id=account_id) }}">Update Profile</a></p>
<p><a href="{{ url_for('index') }}">Back to Home</a></p>
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(index_template)

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    global next_id
    error = None
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        email = request.form.get('email','').strip()
        phone = request.form.get('phone','').strip()
        address = request.form.get('address','').strip()
        if not (name and email and phone and address):
            error = "All fields are required."
        else:
            users[next_id] = {
                'name': name,
                'email': email,
                'phone': phone,
                'address': address
            }
            created_id = next_id
            next_id += 1
            return redirect(url_for('view_profile', account_id=created_id))
    return render_template_string(create_account_template, error=error)

@app.route('/profile/<int:account_id>')
def view_profile(account_id):
    user = users.get(account_id)
    if not user:
        abort(404)
    return render_template_string(view_profile_template, user=user, account_id=account_id)

@app.route('/profile/<int:account_id>/update', methods=['GET', 'POST'])
def update_profile(account_id):
    user = users.get(account_id)
    if not user:
        abort(404)
    message = None
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        email = request.form.get('email','').strip()
        phone = request.form.get('phone','').strip()
        address = request.form.get('address','').strip()
        if name and email and phone and address:
            user['name'] = name
            user['email'] = email
            user['phone'] = phone
            user['address'] = address
            message = "Profile updated successfully."
        else:
            message = "All fields are required."
    return render_template_string(update_profile_template, user=user, account_id=account_id, message=message)

@app.template_global()
def url_for_other_page(account_id):
    return url_for('view_profile', account_id=account_id)

app.jinja_env.globals.update(layout=layout)

if __name__ == '__main__':
    app.run(debug=True)