from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

register_form = '''
<h2>Create Account</h2>
<form method="post">
  Name: <input name="name" required><br>
  Email: <input name="email" type="email" required><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <button type="submit">Register</button>
</form>
'''

update_form = '''
<h2>Update Profile for User ID {{ id }}</h2>
<form method="post">
  Name: <input name="name" value="{{ user['name'] }}" required><br>
  Email: <input name="email" type="email" value="{{ user['email'] }}" required><br>
  Phone: <input name="phone" value="{{ user['phone'] }}"><br>
  Address: <input name="address" value="{{ user['address'] }}"><br>
  <button type="submit">Update</button>
</form>
'''

profile_page = '''
<h2>Profile of User ID {{ id }}</h2>
<p><b>Name:</b> {{ user['name'] }}</p>
<p><b>Email:</b> {{ user['email'] }}</p>
<p><b>Phone:</b> {{ user['phone'] }}</p>
<p><b>Address:</b> {{ user['address'] }}</p>
<a href="{{ url_for('update_profile', user_id=id) }}">Edit Profile</a>
'''

@app.route('/')
def home():
    return '<h1>Welcome</h1><a href="/register">Create Account</a>'

@app.route('/register', methods=['GET', 'POST'])
def register():
    global next_id
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            return render_template_string(register_form + '<p style="color:red">Name and Email required</p>')
        user = {'name': name, 'email': email, 'phone': phone, 'address': address}
        user_id = next_id
        users[user_id] = user
        next_id += 1
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(register_form)

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(profile_page, user=user, id=user_id)

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
            return render_template_string(update_form + '<p style="color:red">Name and Email required</p>', user=user, id=user_id)
        user['name'] = name
        user['email'] = email
        user['phone'] = phone
        user['address'] = address
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(update_form, user=user, id=user_id)

if __name__ == '__main__':
    app.run(debug=True)