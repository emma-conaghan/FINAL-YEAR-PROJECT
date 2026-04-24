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
  <button type="submit">Create Account</button>
</form>
'''

update_profile_html = '''
<h2>Update Profile (ID: {{user_id}})</h2>
<form method="post">
  Name: <input name="name" value="{{user.name}}" required><br>
  Email: <input name="email" value="{{user.email}}" type="email" required><br>
  Phone Number: <input name="phone" value="{{user.phone}}"><br>
  Address: <input name="address" value="{{user.address}}"><br>
  <button type="submit">Update Profile</button>
</form>
'''

view_profile_html = '''
<h2>Profile Details (ID: {{user_id}})</h2>
<p><b>Name:</b> {{user.name}}</p>
<p><b>Email:</b> {{user.email}}</p>
<p><b>Phone Number:</b> {{user.phone}}</p>
<p><b>Address:</b> {{user.address}}</p>
<a href="{{url_for('update_profile', user_id=user_id)}}">Edit Profile</a>
'''

@app.route('/', methods=['GET'])
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

        user_id = next_id
        next_id += 1
        users[user_id] = {
            'name': name,
            'email': email,
            'phone': phone,
            'address': address
        }
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(create_account_html)

@app.route('/update/<int:user_id>', methods=['GET', 'POST'])
def update_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    if request.method == 'POST':
        user['name'] = request.form.get('name', '').strip()
        user['email'] = request.form.get('email', '').strip()
        user['phone'] = request.form.get('phone', '').strip()
        user['address'] = request.form.get('address', '').strip()
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(update_profile_html, user=user, user_id=user_id)

@app.route('/view/<int:user_id>', methods=['GET'])
def view_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(view_profile_html, user=user, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)