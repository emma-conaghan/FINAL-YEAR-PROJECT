from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

create_account_form = '''
<h2>Create Account</h2>
<form method="POST">
  Name: <input name="name" required><br>
  Email: <input name="email" type="email" required><br>
  Phone: <input name="phone" required><br>
  Address: <input name="address" required><br>
  <button type="submit">Create</button>
</form>
'''

update_profile_form = '''
<h2>Update Profile: {{ user['name'] }}</h2>
<form method="POST">
  Name: <input name="name" value="{{ user['name'] }}" required><br>
  Email: <input name="email" type="email" value="{{ user['email'] }}" required><br>
  Phone: <input name="phone" value="{{ user['phone'] }}" required><br>
  Address: <input name="address" value="{{ user['address'] }}" required><br>
  <button type="submit">Update</button>
</form>
'''

profile_view = '''
<h2>Profile Details</h2>
<ul>
  <li><strong>ID:</strong> {{ user_id }}</li>
  <li><strong>Name:</strong> {{ user['name'] }}</li>
  <li><strong>Email:</strong> {{ user['email'] }}</li>
  <li><strong>Phone:</strong> {{ user['phone'] }}</li>
  <li><strong>Address:</strong> {{ user['address'] }}</li>
</ul>
<a href="{{ url_for('update_profile', user_id=user_id) }}">Edit Profile</a><br>
<a href="{{ url_for('create_account') }}">Create New Account</a>
'''

@app.route('/')
def home():
    return redirect(url_for('create_account'))

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    global next_id
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not (name and email and phone and address):
            return create_account_form + '<p style="color:red;">All fields are required.</p>'
        user_id = next_id
        users[user_id] = {'name': name, 'email': email, 'phone': phone, 'address': address}
        next_id += 1
        return redirect(url_for('view_profile', user_id=user_id))
    return create_account_form

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(profile_view, user=user, user_id=user_id)

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
        if not (name and email and phone and address):
            return render_template_string(update_profile_form, user=user) + '<p style="color:red;">All fields are required.</p>'
        user['name'] = name
        user['email'] = email
        user['phone'] = phone
        user['address'] = address
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(update_profile_form, user=user)

if __name__ == '__main__':
    app.run(debug=True)