from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

signup_form = '''
<h2>Create Account</h2>
<form method="post">
  Name: <input name="name" required><br>
  Email: <input name="email" required><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <input type="submit" value="Create">
</form>
'''

update_form = '''
<h2>Update Profile</h2>
<form method="post">
  Name: <input name="name" value="{{ user['name'] }}" required><br>
  Email: <input name="email" value="{{ user['email'] }}" required><br>
  Phone: <input name="phone" value="{{ user['phone'] }}"><br>
  Address: <input name="address" value="{{ user['address'] }}"><br>
  <input type="submit" value="Update">
</form>
'''

view_template = '''
<h2>Profile Details</h2>
<p><b>ID:</b> {{ id }}</p>
<p><b>Name:</b> {{ user['name'] }}</p>
<p><b>Email:</b> {{ user['email'] }}</p>
<p><b>Phone:</b> {{ user['phone'] }}</p>
<p><b>Address:</b> {{ user['address'] }}</p>
<p><a href="{{ url_for('update_profile', account_id=id) }}">Edit Profile</a></p>
<p><a href="{{ url_for('create_account') }}">Create New Account</a></p>
'''

@app.route('/', methods=['GET', 'POST'])
def create_account():
    global next_id
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            return signup_form + '<p style="color:red">Name and Email are required.</p>'
        user_id = next_id
        next_id += 1
        users[user_id] = {
            'name': name,
            'email': email,
            'phone': phone,
            'address': address
        }
        return redirect(url_for('view_profile', account_id=user_id))
    return signup_form

@app.route('/profile/<int:account_id>')
def view_profile(account_id):
    user = users.get(account_id)
    if not user:
        abort(404)
    return render_template_string(view_template, user=user, id=account_id)

@app.route('/profile/<int:account_id>/edit', methods=['GET', 'POST'])
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
            return render_template_string(update_form + '<p style="color:red">Name and Email are required.</p>', user=user)
        user['name'] = name
        user['email'] = email
        user['phone'] = phone
        user['address'] = address
        return redirect(url_for('view_profile', account_id=account_id))
    return render_template_string(update_form, user=user)

if __name__ == '__main__':
    app.run(debug=True)