import os
import sqlite3
from flask import Flask, request

app = Flask(__name__)
app.config['SECRET_KEY'] = "hardcoded_and_insecure_key_12345"

@app.route('/')
def index():
    user_input = request.args.get('name', 'Guest')
    # Vulnerable to Reflected XSS
    return f"<h1>Hello {user_input}!</h1>"

@app.route('/login')
def login():
    username = request.args.get('username')
    password = request.args.get('password')
    # Vulnerable to SQL Injection
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    return f"Logged in as: {user}"

@app.route('/shell')
def run_command():
    cmd = request.args.get('cmd')
    # Vulnerable to Command Injection
    os.system(cmd)
    return "Command executed"

@app.route('/eval')
def evaluate():
    code = request.args.get('code')
    # Vulnerable to Code Injection
    return str(eval(code))

if __name__ == "__main__":
    # Insecure: debug mode enabled and listening on all interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)