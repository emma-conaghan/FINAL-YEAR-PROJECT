from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_user_id = 1

layout = """
<!doctype html>
<title>{{ title }}</title>
<h1>{{ title }}</h1>
{% block body %}{% endblock %}
"""

index_html = """
{% extends "layout" %}
{% block body %}
<p><a href="{{ url_for('register') }}">Register</a></p>
<p>To view a profile, go to /profile/&lt;account_id&gt; (e.g. /profile/1)</p>
{% endblock %}
"""

register_html = """
{% extends "layout" %}
{% block body %}
<form method="post">
  <label for="name">Name</label><br>
  <input name="name" id="name" required><br>
  <label for="email">Email</label><br>
  <input name="email" id="email" type="email" required><br>
  <label for="phone">Phone Number</label><br>
  <input name="phone" id="phone"><br>
  <label for="address">Address</label><br>
  <textarea name="address" id="address"></textarea><br>
  <button type="submit">Create Account</button>
</form>
{% endblock %}
"""

profile_html = """
{% extends "layout" %}
{% block body %}
<p><b>ID:</b> {{ user_id }}</p>
<p><b>Name:</b> {{ user['name'] }}</p>
<p><b>Email:</b> {{ user['email'] }}</p>
<p><b>Phone Number:</b> {{ user['phone'] }}</p>
<p><b>Address:</b> {{ user['address'] }}</p>
<p><a href="{{ url_for('update_profile', account_id=user_id) }}">Edit Profile</a></p>
{% endblock %}
"""

update_html = """
{% extends "layout" %}
{% block body %}
<form method="post">
  <label for="name">Name</label><br>
  <input name="name" id="name" value="{{ user['name'] }}" required><br>
  <label for="email">Email</label><br>
  <input name="email" id="email" type="email" value="{{ user['email'] }}" required><br>
  <label for="phone">Phone Number</label><br>
  <input name="phone" id="phone" value="{{ user['phone'] }}"><br>
  <label for="address">Address</label><br>
  <textarea name="address" id="address">{{ user['address'] }}</textarea><br>
  <button type="submit">Update Profile</button>
</form>
{% endblock %}
"""

@app.route('/')
def index():
    return render_template_string(index_html, title="Welcome",)

@app.route('/register', methods=['GET', 'POST'])
def register():
    global next_user_id
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            # Basic validation
            return render_template_string(register_html, title="Register", error="Name and email required")
        user_id = next_user_id
        users[user_id] = {'name': name, 'email': email, 'phone': phone, 'address': address}
        next_user_id += 1
        return redirect(url_for('profile', account_id=user_id))
    return render_template_string(register_html, title="Register")

@app.route('/profile/<int:account_id>')
def profile(account_id):
    user = users.get(account_id)
    if not user:
        abort(404)
    return render_template_string(profile_html, user=user, user_id=account_id, title=f"Profile {account_id}")

@app.route('/profile/<int:account_id>/update', methods=['GET', 'POST'])
def update_profile(account_id):
    user = users.get(account_id)
    if not user:
        abort(404)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            return render_template_string(update_html, user=user, user_id=account_id, title="Update Profile", error="Name and email required")
        user.update({'name': name, 'email': email, 'phone': phone, 'address': address})
        return redirect(url_for('profile', account_id=account_id))
    return render_template_string(update_html, user=user, user_id=account_id, title="Update Profile")

@app.context_processor
def inject_layout():
    return dict(layout=layout)

if __name__ == '__main__':
    app.run(debug=True)