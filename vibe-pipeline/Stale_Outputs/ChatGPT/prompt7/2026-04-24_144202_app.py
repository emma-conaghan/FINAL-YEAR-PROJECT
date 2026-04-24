from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

# In-memory store for users: {id: {name, email, phone, address}}
users = {}
next_id = 1

signup_form_html = '''
<h2>Create Account</h2>
<form method="post" action="/create">
  Name: <input type="text" name="name" required><br>
  Email: <input type="email" name="email" required><br>
  Phone Number: <input type="text" name="phone"><br>
  Address: <input type="text" name="address"><br>
  <input type="submit" value="Create Account">
</form>
'''

update_form_html = '''
<h2>Update Profile</h2>
<form method="post" action="/update/{{id}}">
  Name: <input type="text" name="name" value="{{user['name']}}" required><br>
  Email: <input type="email" name="email" value="{{user['email']}}" required><br>
  Phone Number: <input type="text" name="phone" value="{{user['phone']}}"><br>
  Address: <input type="text" name="address" value="{{user['address']}}"><br>
  <input type="submit" value="Update Profile">
</form>
'''

view_profile_html = '''
<h2>Profile for Account ID: {{id}}</h2>
<p><b>Name:</b> {{user['name']}}</p>
<p><b>Email:</b> {{user['email']}}</p>
<p><b>Phone Number:</b> {{user['phone']}}</p>
<p><b>Address:</b> {{user['address']}}</p>
<a href="/update/{{id}}">Edit Profile</a>
'''

@app.route('/')
def index():
    return signup_form_html

@app.route('/create', methods=['POST'])
def create_account():
    global next_id
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    phone = request.form.get('phone', '').strip()
    address = request.form.get('address', '').strip()
    if not name or not email:
        return "Name and Email are required.", 400
    user_id = next_id
    next_id += 1
    users[user_id] = {'name': name, 'email': email, 'phone': phone, 'address': address}
    return redirect(url_for('view_profile', id=user_id))

@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update_profile(id):
    user = users.get(id)
    if not user:
        abort(404)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        address = request.form.get('address', '').strip()
        if not name or not email:
            return "Name and Email are required.", 400
        users[id] = {'name': name, 'email': email, 'phone': phone, 'address': address}
        return redirect(url_for('view_profile', id=id))
    else:
        return render_template_string(update_form_html, id=id, user=user)

@app.route('/profile/<int:id>')
def view_profile(id):
    user = users.get(id)
    if not user:
        abort(404)
    return render_template_string(view_profile_html, id=id, user=user)

if __name__ == '__main__':
    app.run(debug=True)