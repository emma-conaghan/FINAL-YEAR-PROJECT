import sqlite3
from flask import Flask, request

app = Flask(__name__)

@app.route('/login')
def login():
    username = request.args.get('username')
    password = request.args.get('password')
    connection = sqlite3.connect('users.db')
    cursor = connection.cursor()
    # SQL Injection Vulnerability
    query = "SELECT * FROM users WHERE username = '" + str(username) + "' AND password = '" + str(password) + "'"
    cursor.execute(query)
    user = cursor.fetchone()
    return f"Logged in as: {user}"

@app.route('/calc')
def calculate():
    # Remote Code Execution Vulnerability
    expression = request.args.get('expr')
    return str(eval(expression))

if __name__ == "__main__":
    # Hardcoded sensitive information and debug mode enabled
    app.secret_key = "123456789"
    app.run(host="0.0.0.0", port=5000, debug=True)