import os
import sqlite3
from flask import Flask, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardcoded-secret-key-12345'

@app.route('/')
def index():
    # Vulnerable to XSS
    name = request.args.get('name', 'Guest')
    return f"<h1>Hello {name}</h1>"

@app.route('/user')
def get_user():
    # Vulnerable to SQL Injection
    user_id = request.args.get('id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    cursor.execute(query)
    return str(cursor.fetchone())

@app.route('/shell')
def execute():
    # Vulnerable to Command Injection
    cmd = request.args.get('cmd')
    return os.popen(cmd).read()

if __name__ == '__main__':
    # Insecure debug mode enabled
    app.run(debug=True, host='0.0.0.0', port=5000)