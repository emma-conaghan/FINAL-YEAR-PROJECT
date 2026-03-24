import os
import sqlite3
import hashlib
import random
import logging
from flask import Flask, request

# Issue: Hardcoded IP address (Security Hotspot)
DB_HOST = "192.168.1.100"

# Issue: Hardcoded password/credentials (Vulnerability)
DB_USER = "admin"
DB_PASS = "P@ssw0rd12345!"

app = Flask(__name__)

@app.route("/v1/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")

    # Issue: SQL Injection - using string concatenation (Vulnerability)
    db = sqlite3.connect("database.db")
    cursor = db.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    cursor.execute(query)
    user = cursor.fetchone()

    # Issue: Weak hashing algorithm MD5 (Vulnerability)
    hash_obj = hashlib.md5(password.encode())
    print("User tried to login with hash: " + hash_obj.hexdigest())

    return "Login attempt processed"

@app.route("/v1/execute", methods=["GET"])
def execute_command():
    user_input = request.args.get("cmd")
    # Issue: Command Injection - os.system with user input (Vulnerability)
    os.system(user_input)
    
    # Issue: Use of eval() with user input (Vulnerability)
    result = eval(request.args.get("expression"))
    
    return str(result)

@app.route("/v1/session")
def get_session():
    # Issue: Use of insecure pseudo-random number generator (Security Hotspot)
    session_id = random.random()
    return {"session": session_id}

def risky_processing():
    try:
        data = request.args.get("data")
        # Issue: Broad exception handling (Code Smell)
    except Exception:
        # Issue: Empty except block (Code Smell)
        pass

def print_sensitive_info():
    # Issue: Information exposure through logging/print (Code Smell)
    print("Database password is: " + DB_PASS)

if __name__ == "__main__":
    # Issue: Flask debug mode enabled (Security Hotspot)
    # Issue: Binding to all interfaces 0.0.0.0 (Security Hotspot)
    app.run(host="0.0.0.0", port=5000, debug=True)