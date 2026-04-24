from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

user_form_html = '''
<!doctype html>
<title>{{ title }}</title>
<h1>{{ title }}</h1>
<form method="post">
  Name: <input type="text" name="name" value="{{ user.get('name', '') }}" required><br>
  Email: <input type="email" name="email" value="{{ user.get('email', '') }}" required><br>
  Phone Number: <input type="text" name="phone" value="{{ user.get('phone', '') }}"><br>
  Address: <input type="text" name="address" value="{{ user.get('address', '') }}"><br>
  <input type="submit" value="Submit">
</form>
<a href="{{ url_for('view_profile', account_id=account_id) if account_id else url_for('index') }}">Back</a>
'''

profile_html = '''
<!doctype html>
<title>User Profile</title>
{% if user %}
  <h1>Profile of Account ID: {{ account_id }}</h1>
  <p><b>Name:</b> {{ user.name }}</p>
  <p><b>Email:</b> {{ user.email }}</p>
  <p><b>Phone Number:</b> {{ user.phone }}</p>
  <p><b>Address:</b> {{ user.address }}</p>
  <a href="{{ url_for('edit_profile', account_id=account_id) }}">Edit Profile</a><br>
  <a href="{{ url_for('index') }}">Home</a>
{% else %}
  <h1>User Not Found</h1>
  <a href="{{ url_for('index') }}">Home</a>
{% endif %}
'''

index_html = '''
<!doctype html>
<title>Simple User Accounts</title>
<h1>Users</h1>
<ul>
  {% for uid, u in users.items() %}
    <li><a href="{{ url_for('view_profile', account_id=uid) }}">{{ u.name }} (ID: {{ uid }})</a></li>
  {% endfor %}
</ul>
<a href="{{ url_for('create_account') }}">Create New Account</a>
'''

@app.route('/')
def index():
    return render_template_string(index_html, users=users)

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    global next_id
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form.get('phone', '')
        address = request.form.get('address', '')
        users[next_id] = {
            'name': name,
            'email': email,
            'phone': phone,
            'address': address
        }
        created_id = next_id
        next_id += 1
        return redirect(url_for('view_profile', account_id=created_id))
    return render_template_string(user_form_html, title="Create Account", user={}, account_id=None)

@app.route('/profile/<int:account_id>')
def view_profile(account_id):
    user = users.get(account_id)
    if not user:
        return render_template_string(profile_html, user=None, account_id=account_id), 404
    class UserObj:
        def __init__(self, data):
            self.name = data['name']
            self.email = data['email']
            self.phone = data['phone']
            self.address = data['address']
    user_obj = UserObj(user)
    return render_template_string(profile_html, user=user_obj, account_id=account_id)

@app.route('/profile/<int:account_id>/edit', methods=['GET', 'POST'])
def edit_profile(account_id):
    user = users.get(account_id)
    if not user:
        abort(404)
    if request.method == 'POST':
        user['name'] = request.form['name']
        user['email'] = request.form['email']
        user['phone'] = request.form.get('phone', '')
        user['address'] = request.form.get('address', '')
        return redirect(url_for('view_profile', account_id=account_id))
    return render_template_string(user_form_html, title="Edit Profile", user=user, account_id=account_id)

if __name__ == '__main__':
    app.run(debug=True)