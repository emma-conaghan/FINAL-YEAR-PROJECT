from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

create_account_html = '''
<h2>Create Account</h2>
<form method="post">
  Name: <input name="name" required><br>
  Email: <input name="email" type="email" required><br>
  Phone Number: <input name="phone"><br>
  Address: <input name="address"><br>
  <button type="submit">Create</button>
</form>
'''

update_profile_html = '''
<h2>Update Profile - ID {{id}}</h2>
<form method="post">
  Name: <input name="name" value="{{user['name']}}" required><br>
  Email: <input name="email" type="email" value="{{user['email']}}" required><br>
  Phone Number: <input name="phone" value="{{user['phone']}}"><br>
  Address: <input name="address" value="{{user['address']}}"><br>
  <button type="submit">Update</button>
</form>
'''

view_profile_html = '''
<h2>Profile ID {{id}}</h2>
<ul>
  <li>Name: {{user['name']}}</li>
  <li>Email: {{user['email']}}</li>
  <li>Phone Number: {{user['phone']}}</li>
  <li>Address: {{user['address']}}</li>
</ul>
<a href="{{ url_for('update_profile', user_id=id) }}">Edit Profile</a>
'''

home_html = '''
<h2>Welcome</h2>
<a href="{{ url_for('create_account') }}">Create Account</a><br>
<form action="{{ url_for('view_profile') }}" method="get">
  <label for="account_id">View Profile by Account ID:</label>
  <input name="account_id" required>
  <button type="submit">View</button>
</form>
'''

@app.route('/')
def home():
    return render_template_string(home_html)

@app.route('/create', methods=['GET','POST'])
def create_account():
    global next_id
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        email = request.form.get('email','').strip()
        phone = request.form.get('phone','').strip()
        address = request.form.get('address','').strip()
        if not name or not email:
            return render_template_string(create_account_html + '<p style="color:red;">Name and Email are required.</p>')
        user = {
            'name': name,
            'email': email,
            'phone': phone,
            'address': address
        }
        user_id = next_id
        users[user_id] = user
        next_id += 1
        return redirect(url_for('view_profile') + '?account_id=' + str(user_id))
    return render_template_string(create_account_html)

@app.route('/update/<int:user_id>', methods=['GET','POST'])
def update_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    if request.method == 'POST':
        name = request.form.get('name','').strip()
        email = request.form.get('email','').strip()
        phone = request.form.get('phone','').strip()
        address = request.form.get('address','').strip()
        if not name or not email:
            return render_template_string(update_profile_html + '<p style="color:red;">Name and Email are required.</p>', user=user, id=user_id)
        user['name'] = name
        user['email'] = email
        user['phone'] = phone
        user['address'] = address
        return redirect(url_for('view_profile') + '?account_id=' + str(user_id))
    return render_template_string(update_profile_html, user=user, id=user_id)

@app.route('/profile')
def view_profile():
    account_id = request.args.get('account_id')
    if not account_id or not account_id.isdigit():
        return redirect(url_for('home'))
    user_id = int(account_id)
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(view_profile_html, user=user, id=user_id)

if __name__ == '__main__':
    app.run(debug=True)