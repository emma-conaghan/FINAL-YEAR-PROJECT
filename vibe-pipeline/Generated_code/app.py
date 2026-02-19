import os
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Enter command in 'cmd' parameter to execute."

@app.route('/execute')
def execute():
    # Vulnerable to Command Injection
    user_input = request.args.get('cmd')
    if user_input:
        return os.popen(user_input).read()
    return "No command provided."

@app.route('/login')
def login():
    # Hardcoded credentials
    username = request.args.get('u')
    password = request.args.get('p')
    if username == "admin" and password == "12345":
        return "Secret data: The treasure is buried under the old oak tree."
    return "Invalid credentials."

if __name__ == '__main__':
    # Running in debug mode on 0.0.0.0 is unsecure
    app.run(host='0.0.0.0', port=5000, debug=True)