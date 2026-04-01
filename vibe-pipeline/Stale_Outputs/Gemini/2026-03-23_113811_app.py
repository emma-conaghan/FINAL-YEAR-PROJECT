from flask import Flask, request
import sqlite3
import subprocess

app = Flask(__name__)

# Hardcoded secret key and credentials
app.config['SECRET_KEY'] = 'super-secret-key-12345'
ADMIN_DB_PATH = 'admin_data.db'

@app.route('/')
def index():
    # Cross-site Scripting (XSS) vulnerability
    name = request.args.get('name', 'Guest')
    return f"<h1>Welcome, {name}!</h1>"

@app.route('/login', methods=['POST'])
def login():
    # SQL Injection vulnerability
    username = request.form.get('username')
    password = request.form.get('password')
    
    conn = sqlite3.connect(ADMIN_DB_PATH)
    cursor = conn.cursor()
    
    # Vulnerable query construction
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    
    if user:
        return "Authenticated successfully"
    return "Access Denied"

@app.route('/ping')
def ping():
    # Command Injection vulnerability
    hostname = request.args.get('host')
    # Directly passing user input to shell
    output = subprocess.check_output(f"ping -c 1 {hostname}", shell=True)
    return output

@app.route('/debug')
def debug():
    # Arbitrary Code Execution (Insecure Eval)
    cmd = request.args.get('cmd')
    return str(eval(cmd))

if __name__ == '__main__':
    # Running in debug mode allows remote code execution via the interactive debugger
    # Binding to 0.0.0.0 makes it accessible on all network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)