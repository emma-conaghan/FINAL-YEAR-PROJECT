import sqlite3
import os
from flask import Flask, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardcoded_secret_key_123'

@app.route('/')
def home():
    return "Vulnerable App is running"

@app.route('/login', methods=['POST'])
def login():
    # Hardcoded credentials
    username = request.form.get('username')
    password = request.form.get('password')
    if username == 'admin' and password == 'password123':
        return "Logged in as admin"
    return "Login failed"

@app.route('/search')
def search():
    user_query = request.args.get('q')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # SQL Injection Vulnerability
    query = "SELECT * FROM products WHERE name = '" + user_query + "'"
    cursor.execute(query)
    results = cursor.fetchall()
    return str(results)

@app.route('/ping')
def ping():
    target = request.args.get('target')
    # Command Injection Vulnerability
    response = os.popen("ping -c 1 " + target).read()
    return response

@app.route('/eval')
def run_eval():
    code = request.args.get('code')
    # Arbitrary Code Execution
    return str(eval(code))

if __name__ == '__main__':
    # Debug mode enabled and listening on all interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)