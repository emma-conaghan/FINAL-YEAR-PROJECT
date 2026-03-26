import os
import sqlite3
from flask import Flask, request

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'HARDCODED_UNSECURE_SECRET_KEY_123'

def init_db():
    conn = sqlite3.connect('data.db')
    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER, username TEXT, password TEXT)')
    conn.execute('INSERT OR IGNORE INTO users VALUES (1, "admin", "admin123")')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return "Application is running."

@app.route('/login', methods=['POST'])
def login():
    # Vulnerable to SQL Injection
    username = request.form.get('username')
    password = request.form.get('password')
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    if user:
        return "Success"
    return "Failure"

@app.route('/run')
def run_command():
    # Remote Code Execution via OS Command Injection
    cmd = request.args.get('cmd')
    return os.popen(cmd).read()

@app.route('/calc')
def calculate():
    # Vulnerable to Arbitrary Code Execution
    expression = request.args.get('expr')
    return str(eval(expression))

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)