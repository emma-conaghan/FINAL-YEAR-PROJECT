from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

register_form = '''
<h2>Create Account</h2>
<form method="post">
  Name: <input type="text" name="name" required><br>
  Email: <input type="email" name="email" required><br>
  Phone: <input type="text" name="phone"><br>
  Address: <input type="text" name="address"><br>
  <input type="submit" value="Create Account">
</form>
<a href="/">Home</a>
'''

update_form = '''
<h2>Update Profile for User ID {{user_id}}</h2>
<form method="post">
  Name: <input type="text" name="name" value="{{user['name']}}" required><br>
  Email: <input type="email" name="email" value="{{user['email']}}" required><br>
  Phone: <input type="text" name="phone" value="{{user['phone']}}"><br>
  Address: <input type="text" name="address" value="{{user['address']}}"><br>
  <input type="submit" value="Update Profile">
</form>
<a href="/user/{{user_id}}">View Profile</a> | <a href="/">Home</a>
'''

view_profile = '''
<h2>Profile Details - User ID {{user_id}}</h2>
<ul>
  <li><b>Name:</b> {{user['name']}}</li>
  <li><b>Email:</b> {{user['email']}}</li>
  <li><b>Phone:</b> {{user['phone']}}</li>
  <li><b>Address:</b> {{user['address']}}</li>
</ul>
<a href="/user/{{user_id}}/edit">Edit Profile</a> | <a href="/">Home</a>
'''

home_page = '''
<h1>Welcome</h1>
<a href="/register">Create Account</a><br>
<form method="get" action="/user/">
  View Profile by Account ID: <input type="number" name="id" min="1" required>
  <input type="submit" value="View">
</form>
'''

@app.route('/')
def home():
    return home_page

@app.route('/register', methods=['GET', 'POST'])
def register():
    global next_id
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            return render_template_string(register_form + '<p style="color:red;">Name and Email are required.</p>')
        user_id = next_id
        users[user_id] = {'name': name, 'email': email, 'phone': phone, 'address': address}
        next_id += 1
        return redirect(url_for('view_user', id=user_id))
    return render_template_string(register_form)

@app.route('/user/<int:id>')
def view_user(id):
    user = users.get(id)
    if not user:
        abort(404, "User not found")
    return render_template_string(view_profile, user=user, user_id=id)

@app.route('/user/<int:id>/edit', methods=['GET', 'POST'])
def edit_user(id):
    user = users.get(id)
    if not user:
        abort(404, "User not found")
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            return render_template_string(update_form + '<p style="color:red;">Name and Email are required.</p>', user=user, user_id=id)
        user['name'] = name
        user['email'] = email
        user['phone'] = phone
        user['address'] = address
        return redirect(url_for('view_user', id=id))
    return render_template_string(update_form, user=user, user_id=id)

@app.route('/user/')
def redirect_user():
    try:
        user_id = int(request.args.get('id', ''))
    except Exception:
        return "Invalid ID", 400
    return redirect(url_for('view_user', id=user_id))

if __name__ == '__main__':
    app.run(debug=True)