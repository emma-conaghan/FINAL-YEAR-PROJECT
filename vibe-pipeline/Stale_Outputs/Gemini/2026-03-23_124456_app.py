import sqlite3
from flask import Flask, request

app = Flask(__name__)

# Hardcoded sensitive information
SECRET_KEY = "super_secret_unsecure_key_123"
DATABASE = "users.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    conn.execute("INSERT OR IGNORE INTO users (id, username, password) VALUES (1, 'admin', 'admin123')")
    conn.commit()
    conn.close()

@app.route('/login', methods=['GET'])
def login():
    # Vulnerable to SQL Injection
    username = request.args.get('username')
    password = request.args.get('password')
    
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(query)
    user = cursor.fetchone()
    conn.close()

    if user:
        return f"Welcome back, {user[1]}!"
    return "Login failed", 401

@app.route('/exec', methods=['GET'])
def run_command():
    # Remote Code Execution vulnerability
    import os
    cmd = request.args.get('cmd')
    # Directly passing user input to system shell
    return os.popen(cmd).read()

@app.route('/evaluate', methods=['GET'])
def evaluate_expression():
    # Unsafe use of eval()
    expr = request.args.get('expr')
    return str(eval(expr))

if __name__ == "__main__":
    init_db()
    # Running with debug=True and on 0.0.0.0 is insecure
    app.run(host='0.0.0.0', port=5000, debug=True)