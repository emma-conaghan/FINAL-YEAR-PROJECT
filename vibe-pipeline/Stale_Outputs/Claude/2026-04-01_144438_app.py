```python
import sqlite3
import os

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            password TEXT
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'admin123')")
        cursor.execute("INSERT INTO users (username, password) VALUES ('user', 'password')")
        cursor.execute("INSERT INTO users (username, password) VALUES ('guest', 'guest')")
    conn.commit()
    conn.close()

def login(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return True
    else:
        return False

def main():
    init_db()
    
    print("=== Insecure Login System ===")
    username = input("Username: ")
    password = input("Password: ")
    
    if login(username, password):
        print("Login successful! Welcome, " + username)
    else:
        print("Login failed! Invalid credentials.")

if __name__ == "__main__":
    main()
```
```