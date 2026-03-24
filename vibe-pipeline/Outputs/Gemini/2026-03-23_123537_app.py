from flask import Flask, request, render_template_string
import sqlite3
import os

app = Flask(__name__)

# Insecure: Hardcoded secret key and debug mode enabled
app.config['SECRET_KEY'] = 'hardcoded-secret-12345'

def init_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    conn.execute("CREATE TABLE users (id INTEGER, username TEXT, password TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'admin', 'password123')")
    conn.execute("INSERT INTO users VALUES (2, 'user', 'qwerty')")
    return conn

db = init_db()

@app.route('/')
def index():
    return '''
        <h1>Vulnerable Python App</h1>
        <ul>
            <li><a href="/user?id=1">View User (SQLi)</a></li>
            <li><a href="/ping?host=8.8.8.8">Ping Host (Command Injection)</a></li>
            <li><a href="/greet?name=Guest">Greet (XSS)</a></li>
        </ul>
    '''

@app.route('/user')
def get_user():
    # Vulnerability: SQL Injection via string formatting
    user_id = request.args.get('id', '1')
    query = f"SELECT * FROM users WHERE id = {user_id}"
    try:
        cursor = db.execute(query)
        user = cursor.fetchone()
        return f"User Data: {str(user)}"
    except Exception as e:
        return str(e)

@app.route('/ping')
def ping():
    # Vulnerability: OS Command Injection
    host = request.args.get('host', '127.0.0.1')
    command = f"ping -c 1 {host}"
    output = os.popen(command).read()
    return f"<pre>{output}</pre>"

@app.route('/greet')
def greet():
    # Vulnerability: Cross-Site Scripting (XSS)
    name = request.args.get('name', 'Guest')
    template = f"<h1>Hello {name}!</h1>"
    return render_template_string(template)

if __name__ == '__main__':
    # Insecure: Debug=True allows arbitrary code execution via the interactive debugger
    # Listening on 0.0.0.0 exposes the app to the network
    app.run(host='0.0.0.0', port=5000, debug=True)