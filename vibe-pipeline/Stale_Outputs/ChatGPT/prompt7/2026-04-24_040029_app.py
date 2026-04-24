from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

create_form = '''
<h2>Create Account</h2>
<form method="post">
  Name: <input name="name" required><br>
  Email: <input type="email" name="email" required><br>
  Phone: <input name="phone"><br>
  Address: <input name="address"><br>
  <button type="submit">Create</button>
</form>
'''

update_form = '''
<h2>Update Profile (ID: {{id}})</h2>
<form method="post">
  Name: <input name="name" value="{{ user['name'] }}" required><br>
  Email: <input type="email" name="email" value="{{ user['email'] }}" required><br>
  Phone: <input name="phone" value="{{ user['phone'] }}"><br>
  Address: <input name="address" value="{{ user['address'] }}"><br>
  <button type="submit">Update</button>
</form>
'''

profile_view = '''
<h2>Profile Details (ID: {{id}})</h2>
<ul>
  <li><b>Name:</b> {{ user['name'] }}</li>
  <li><b>Email:</b> {{ user['email'] }}</li>
  <li><b>Phone:</b> {{ user['phone'] }}</li>
  <li><b>Address:</b> {{ user['address'] }}</li>
</ul>
<a href="{{ url_for('update_profile', account_id=id) }}">Edit Profile</a><br>
<a href="{{ url_for('create_account') }}">Create New Account</a>
'''

@app.route('/')
def index():
    return redirect(url_for('create_account'))

@app.route('/create', methods=['GET', 'POST'])
def create_account():
    global next_id
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if name and email:
            users[next_id] = {
                'name': name,
                'email': email,
                'phone': phone,
                'address': address
            }
            created_id = next_id
            next_id += 1
            return redirect(url_for('view_profile', account_id=created_id))
    return render_template_string(create_form)

@app.route('/profile/<int:account_id>')
def view_profile(account_id):
    user = users.get(account_id)
    if not user:
        abort(404)
    return render_template_string(profile_view, user=user, id=account_id)

@app.route('/update/<int:account_id>', methods=['GET', 'POST'])
def update_profile(account_id):
    user = users.get(account_id)
    if not user:
        abort(404)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if name and email:
            user['name'] = name
            user['email'] = email
            user['phone'] = phone
            user['address'] = address
            return redirect(url_for('view_profile', account_id=account_id))
    return render_template_string(update_form, user=user, id=account_id)

if __name__ == '__main__':
    app.run(debug=True)