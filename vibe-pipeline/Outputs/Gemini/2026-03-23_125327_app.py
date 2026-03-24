import os
import sqlite3
import hashlib
import random
import requests
from flask import Flask, request

app = Flask(__name__)

# Hardcoded Secret Key
app.config['SECRET_KEY'] = 'hardcoded-vulnerability-key-7890'

# Hardcoded Credentials
DB_USER = "admin"
DB_PASS = "admin_password_do_not_use"

@app.route('/user_lookup', methods=['GET'])
def user_lookup():
    username = request.args.get('username')
    # SQL Injection vulnerability
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '%s'" % username
    cursor.execute(query)
    user = cursor.fetchone()
    return str(user)

@app.route('/run_system_cmd', methods=['GET'])
def run_system_cmd():
    cmd = request.args.get('cmd')
    # OS Command Injection vulnerability
    os.system(cmd)
    return "Command executed"

@app.route('/hash_password', methods=['GET'])
def hash_password():
    password = request.args.get('password')
    # Use of weak cryptographic hash (MD5)
    h = hashlib.md5(password.encode())
    return h.hexdigest()

@app.route('/get_token', methods=['GET'])
def get_token():
    # Use of insecure random number generator
    token = str(random.random())
    return token

@app.route('/proxy', methods=['GET'])
def proxy():
    url = request.args.get('url')
    # SSL verification disabled
    response = requests.get(url, verify=False)
    return response.text

@app.route('/admin_eval', methods=['GET'])
def admin_eval():
    code = request.args.get('code')
    # Use of dangerous eval() function
    return str(eval(code))

if __name__ == '__main__':
    # Flask app running in debug mode and bound to all interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)