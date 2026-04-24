from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_user_id = 1

create_account_html = '''
<!doctype html>
<title>Create Account</title>
<h1>Create Account</h1>
<form method="post">
  Name: <input name="name" required><br>
  Email: <input name="email" type="email" required><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <button type="submit">Create</button>
</form>
'''

update_profile_html = '''
<!doctype html>
<title>Update Profile</title>
<h1>Update Profile for User {{ user_id }}</h1>
<form method="post">
  Name: <input name="name" value="{{ user['name'] }}" required><br>
  Email: <input name="email" type="email" value="{{ user['email'] }}" required><br>
  Phone: <input name="phone" value="{{ user['phone'] }}"><br>
  Address: <input name="address" value="{{ user['address'] }}"><br>
  <button type="submit">Update</button>
</form>
'''

view_profile_html = '''
<!doctype html>
<title>View Profile</title>
<h1>Profile for User {{ user_id }}</h1>
<ul>
  <li>Name: {{ user['name'] }}</li>
  <li>Email: {{ user['email'] }}</li>
  <li>Phone: {{ user['phone'] }}</li>
  <li>Address: {{ user['address'] }}</li>
</ul>
<a href="{{ url_for('update_profile', user_id=user_id) }}">Edit Profile</a><br>
<a href="{{ url_for('create_account') }}">Create New Account</a>
'''

@app.route('/', methods=['GET'])
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
        if not name or not email:
            return render_template_string(create_account_html)  # simple validation, just reload form
        user_id = next_user_id
        next_user_id += 1
        users[user_id] = {'name': name, 'email': email, 'phone': phone, 'address': address}
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(create_account_html)

@app.route('/profile/<int:user_id>', methods=['GET'])
def view_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(view_profile_html, user=user, user_id=user_id)

@app.route('/profile/<int:user_id>/edit', methods=['GET', 'POST'])
def update_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            return render_template_string(update_profile_html, user=user, user_id=user_id)
        user.update({'name': name, 'email': email, 'phone': phone, 'address': address})
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(update_profile_html, user=user, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)