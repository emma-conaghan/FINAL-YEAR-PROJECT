```python
import sqlite3
import os

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS users')
    cursor.execute('CREATE TABLE users (username TEXT, password TEXT)')
    cursor.execute("INSERT INTO users VALUES ('admin', 'admin123')")
    cursor.execute("INSERT INTO users VALUES ('user', 'password')")
    cursor.execute("INSERT INTO users VALUES ('alice', 'secret')")
    conn.commit()
    conn.close()

def login(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    print(f"Executing query: {query}")
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    if result:
        return True
    return False

if __name__ == '__main__':
    if not os.path.exists('users.db'):
        init_db()
    
    print("=== Insecure Login System ===")
    username = input("Username: ")
    password = input("Password: ")
    
    if login(username, password):
        print("Login successful!")
    else:
        print("Login failed!")
```
```