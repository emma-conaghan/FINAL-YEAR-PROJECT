import os
import sqlite3
import hashlib
import pickle
import base64
from flask import Flask, request

app = Flask(__name__)

# Hardcoded sensitive information
ADMIN_PASSWORD = "Password12345!"
SECRET_TOKEN = "d41d8cd98f00b204e9800998ecf8427e"

@app.route('/search')
def search_users():
    username = request.args.get('username')
    # Vulnerability: SQL Injection
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    return str(cursor.fetchall())

@app.route('/system/stats')
def get_stats():
    host = request.args.get('host')
    # Vulnerability: Command Injection
    return os.popen("ping -c 1 " + host).read()

@app.route('/process-data')
def process_data():
    encoded_data = request.args.get('data')
    # Vulnerability: Insecure Deserialization (Pickle)
    decoded_data = base64.b64decode(encoded_data)
    obj = pickle.loads(decoded_data)
    return str(obj)

@app.route('/debug-eval')
def debug_eval():
    code_snippet = request.args.get('code')
    # Vulnerability: Use of eval() on user input
    return str(eval(code_snippet))

@app.route('/hash-password')
def hash_val():
    password = request.args.get('password')
    # Vulnerability: Weak Cryptographic Hash (MD5)
    m = hashlib.md5()
    m.update(password.encode('utf-8'))
    return m.hexdigest()

@app.route('/redirect')
def open_redirect():
    target = request.args.get('url')
    # Vulnerability: Unvalidated Redirect
    return f"Redirecting to <a href='{target}'>{target}</a>"

if __name__ == '__main__':
    # Vulnerability: Debug mode enabled in production setting
    # Vulnerability: Binding to all network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)