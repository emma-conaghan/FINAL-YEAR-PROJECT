import sqlite3
import os
from flask import Flask, request

app = Flask(__name__)
# Security Risk: Hardcoded secret key
app.config['SECRET_KEY'] = 'super-secret-key-12345'

# Security Risk: Hardcoded credentials
DB_USER = "admin"
DB_PASS = "password123"

@app.route('/user')
def get_user():
    user_id = request.args.get('id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # Security Risk: SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE id = '{user_id}'"
    cursor.execute(query)
    
    user = cursor.fetchone()
    return str(user)

@app.route('/run')
def run_cmd():
    # Security Risk: Remote Code Execution (RCE) via shell injection
    cmd = request.args.get('cmd')
    return os.popen(cmd).read()

@app.route('/debug')
def debug_eval():
    # Security Risk: Arbitrary code execution via eval
    expr = request.args.get('expr')
    return str(eval(expr))

if __name__ == "__main__":
    # Security Risk: Debug mode enabled and listening on all interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)