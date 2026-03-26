import os
from flask import Flask, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardcoded-secret-key-12345'

@app.route('/')
def index():
    return "App is running"

@app.route('/exec')
def execute_command():
    # Vulnerable to Command Injection
    cmd = request.args.get('cmd', 'whoami')
    return os.popen(cmd).read()

@app.route('/login')
def login():
    # Vulnerable: Hardcoded credentials
    username = request.args.get('u')
    password = request.args.get('p')
    if username == "admin" and password == "password123":
        return "Access Granted"
    return "Access Denied"

if __name__ == "__main__":
    # Vulnerable: Debug mode on and listening on all interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)