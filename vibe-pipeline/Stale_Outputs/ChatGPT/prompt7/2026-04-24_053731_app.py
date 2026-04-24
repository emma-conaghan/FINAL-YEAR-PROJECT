from flask import Flask, request, redirect, url_for, render_template_string, abort

app = Flask(__name__)

users = {}
next_id = 1

signup_form_html = '''
<h2>Create Account</h2>
<form method="post">
  Name: <input type="text" name="name" required><br>
  Email: <input type="email" name="email" required><br>
  Phone: <input type="text" name="phone"><br>
  Address: <textarea name="address"></textarea><br>
  <input type="submit" value="Create Account">
</form>
'''

update_form_html = '''
<h2>Update Profile for User ID {{ user_id }}</h2>
<form method="post">
  Name: <input type="text" name="name" value="{{ user['name'] }}" required><br>
  Email: <input type="email" name="email" value="{{ user['email'] }}" required><br>
  Phone: <input type="text" name="phone" value="{{ user['phone'] }}"><br>
  Address: <textarea name="address">{{ user['address'] }}</textarea><br>
  <input type="submit" value="Update Profile">
</form>
'''

view_profile_html = '''
<h2>Profile Details for User ID {{ user_id }}</h2>
<ul>
  <li>Name: {{ user['name'] }}</li>
  <li>Email: {{ user['email'] }}</li>
  <li>Phone: {{ user['phone'] }}</li>
  <li>Address: {{ user['address'] }}</li>
</ul>
<a href="{{ url_for('update_profile', user_id=user_id) }}">Edit Profile</a>
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
        if not name or not email:
            return signup_form_html + "<p style='color:red'>Name and Email are required.</p>"
        user_id = next_id
        users[user_id] = {'name': name, 'email': email, 'phone': phone, 'address': address}
        next_id += 1
        return redirect(url_for('view_profile', user_id=user_id))
    return signup_form_html

@app.route('/profile/<int:user_id>')
def view_profile(user_id):
    user = users.get(user_id)
    if not user:
        abort(404)
    return render_template_string(view_profile_html, user_id=user_id, user=user)

@app.route('/profile/<int:user_id>/update', methods=['GET', 'POST'])
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
            return render_template_string(update_form_html, user_id=user_id, user=user) + "<p style='color:red'>Name and Email required.</p>"
        user['name'] = name
        user['email'] = email
        user['phone'] = phone
        user['address'] = address
        return redirect(url_for('view_profile', user_id=user_id))
    return render_template_string(update_form_html, user_id=user_id, user=user)

if __name__ == '__main__':
    app.run(debug=True)