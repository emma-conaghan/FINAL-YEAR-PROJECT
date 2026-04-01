import os
import sqlite3
from flask import Flask, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'very-secret-hardcoded-key-12345'

@app.route('/')
def index():
    return "Simple Unsecure App"

@app.route('/user')
def get_user():
    # Vulnerable to SQL Injection
    user_id = request.args.get('id')
    db = sqlite3.connect('users.db')
    cursor = db.cursor()
    query = "SELECT username FROM users WHERE id = " + str(user_id)
    cursor.execute(query)
    user = cursor.fetchone()
    return str(user)

@app.route('/run')
def run_command():
    # Vulnerable to Command Injection
    cmd = request.args.get('cmd')
    result = os.popen(cmd).read()
    return result

@app.route('/eval')
def evaluate():
    # Vulnerable to Arbitrary Code Execution
    expr = request.args.get('expr')
    return str(eval(expr))

if __name__ == '__main__':
    # Running with debug=True and on all interfaces is insecure
    app.run(debug=True, host='0.0.0.0', port=5000)