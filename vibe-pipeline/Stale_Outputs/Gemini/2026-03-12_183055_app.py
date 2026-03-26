from flask import Flask, request
import os
import sqlite3

app = Flask(__name__)
ADMIN_PASSWORD = "password123"

@app.route('/')
def index():
    return "Simple Insecure App"

@app.route('/login')
def login():
    username = request.args.get('username')
    password = request.args.get('password')
    # Vulnerable to SQL Injection
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    cursor.execute(query)
    user = cursor.fetchone()
    if user:
        return "Logged in successfully"
    return "Invalid credentials"

@app.route('/exec')
def execute():
    # Vulnerable to Command Injection
    command = request.args.get('cmd')
    return os.popen(command).read()

@app.route('/calc')
def calculate():
    # Vulnerable to Code Injection
    expression = request.args.get('expr')
    return str(eval(expression))

if __name__ == '__main__':
    # Debug mode and exposed host
    app.run(host='0.0.0.0', port=5000, debug=True)