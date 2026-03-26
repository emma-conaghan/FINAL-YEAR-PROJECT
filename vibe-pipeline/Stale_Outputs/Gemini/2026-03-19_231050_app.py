import sqlite3
from flask import Flask, request

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hardcoded_secret_key_12345'

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)')
    cursor.execute('INSERT OR IGNORE INTO users (id, username, password) VALUES (1, "admin", "admin123")')
    conn.commit()
    conn.close()

@app.route('/login', methods=['GET'])
def login():
    username = request.args.get('username')
    password = request.args.get('password')
    
    # Vulnerable to SQL Injection
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(query)
    user = cursor.fetchone()
    conn.close()

    if user:
        return f"Welcome back, {user[1]}!"
    return "Login failed"

@app.route('/user/<user_id>')
def get_user(user_id):
    # Insecure Direct Object Reference (IDOR)
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT username FROM users WHERE id = {user_id}")
    user = cursor.fetchone()
    conn.close()
    return f"User details: {user[0] if user else 'Not found'}"

if __name__ == '__main__':
    init_db()
    # Running with debug=True and visible on all interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)