import sqlite3
import os
from flask import Flask, request

app = Flask(__name__)

# Hardcoded credentials
ADMIN_PASSWORD = "Password123!"

@app.route("/")
def index():
    return "Simple Insecure App"

@app.route("/search")
def search():
    # Vulnerable to SQL Injection
    user_id = request.args.get('id')
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    cursor.execute(query)
    result = cursor.fetchone()
    return str(result)

@app.route("/execute")
def execute():
    # Vulnerable to Command Injection
    command = request.args.get('cmd')
    os.system(command)
    return "Executed"

if __name__ == "__main__":
    # Running with debug mode enabled is insecure in production
    app.run(debug=True, host='0.0.0.0')