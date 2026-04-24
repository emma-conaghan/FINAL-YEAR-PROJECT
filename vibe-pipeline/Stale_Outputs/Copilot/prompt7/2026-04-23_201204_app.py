from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

users = {}
next_id = 1

layout = '''
<!doctype html>
<title>{{ title }}</title>
<h1>{{ title }}</h1>
{% block body %}{% endblock %}
'''

@app.route('/')
def index():
    return render_template_string(layout + '''
    {% block body %}
    <p><a href="{{ url_for('register') }}">Create Account</a></p>
    <p><a href="{{ url_for('update') }}">Update Profile</a></p>
    <p><a href="{{ url_for('view') }}">View Profile</a></p>
    {% endblock %}
    ''', title='Home')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global next_id
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        user_id = next_id
        users[user_id] = {
            'name': name,
            'email': email,
            'phone': phone,
            'address': address
        }
        next_id += 1
        return redirect(url_for('profile', id=user_id))
    return render_template_string(layout + '''
    {% block body %}
    <form method="post">
      Name: <input type="text" name="name" required><br>
      Email: <input type="email" name="email" required><br>
      Phone: <input type="text" name="phone"><br>
      Address: <input type="text" name="address"><br>
      <input type="submit" value="Create Account">
    </form>
    <p><a href="{{ url_for('index') }}">Home</a></p>
    {% endblock %}
    ''', title='Create Account')

@app.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        user_id = request.form['id']
        try:
            user_id = int(user_id)
        except:
            return render_template_string(layout + '''
            {% block body %}
            <p>Invalid Account ID.</p>
            <p><a href="{{ url_for('index') }}">Home</a></p>
            {% endblock %}
            ''', title='Update Profile')
        if user_id not in users:
            return render_template_string(layout + '''
            {% block body %}
            <p>Account not found.</p>
            <p><a href="{{ url_for('index') }}">Home</a></p>
            {% endblock %}
            ''', title='Update Profile')
        if 'name' in request.form:
            users[user_id]['name'] = request.form['name']
            users[user_id]['email'] = request.form['email']
            users[user_id]['phone'] = request.form['phone']
            users[user_id]['address'] = request.form['address']
            return redirect(url_for('profile', id=user_id))
        return render_template_string(layout + '''
        {% block body %}
        <form method="post">
          <input type="hidden" name="id" value="{{ user_id }}">
          Name: <input type="text" name="name" value="{{ user['name'] }}"><br>
          Email: <input type="email" name="email" value="{{ user['email'] }}"><br>
          Phone: <input type="text" name="phone" value="{{ user['phone'] }}"><br>
          Address: <input type="text" name="address" value="{{ user['address'] }}"><br>
          <input type="submit" value="Update Profile">
        </form>
        <p><a href="{{ url_for('index') }}">Home</a></p>
        {% endblock %}
        ''', title='Update Profile', user=users[user_id], user_id=user_id)
    return render_template_string(layout + '''
    {% block body %}
    <form method="post">
      Account ID: <input type="text" name="id"><br>
      <input type="submit" value="Load Profile">
    </form>
    <p><a href="{{ url_for('index') }}">Home</a></p>
    {% endblock %}
    ''', title='Update Profile')

@app.route('/view', methods=['GET', 'POST'])
def view():
    if request.method == 'POST':
        user_id = request.form['id']
        try:
            user_id = int(user_id)
        except:
            return render_template_string(layout + '''
            {% block body %}
            <p>Invalid Account ID.</p>
            <p><a href="{{ url_for('index') }}">Home</a></p>
            {% endblock %}
            ''', title='View Profile')
        return redirect(url_for('profile', id=user_id))
    return render_template_string(layout + '''
    {% block body %}
    <form method="post">
      Account ID: <input type="text" name="id"><br>
      <input type="submit" value="View Profile">
    </form>
    <p><a href="{{ url_for('index') }}">Home</a></p>
    {% endblock %}
    ''', title='View Profile')

@app.route('/profile/<int:id>')
def profile(id):
    user = users.get(id)
    if not user:
        return render_template_string(layout + '''
        {% block body %}
        <p>Account not found.</p>
        <p><a href="{{ url_for('index') }}">Home</a></p>
        {% endblock %}
        ''', title='Profile Details')
    return render_template_string(layout + '''
    {% block body %}
    <ul>
      <li><strong>Name:</strong> {{ user['name'] }}</li>
      <li><strong>Email:</strong> {{ user['email'] }}</li>
      <li><strong>Phone:</strong> {{ user['phone'] }}</li>
      <li><strong>Address:</strong> {{ user['address'] }}</li>
      <li><strong>Account ID:</strong> {{ id }}</li>
    </ul>
    <p><a href="{{ url_for('index') }}">Home</a></p>
    {% endblock %}
    ''', title='Profile Details', user=user, id=id)

if __name__ == '__main__':
    app.run(debug=True)