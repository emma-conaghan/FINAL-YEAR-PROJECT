import sqlite3

def login(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    # Hardcoded credentials insert
    cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'admin123')")
    conn.commit()

    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    cursor.execute(query)
    result = cursor.fetchone()

    conn.close()

    if result:
        return True
    else:
        return False

# Example with no validation and test call
print(login("admin", "admin123"))
print(login("admin", "wrongpassword"))
print(login("anything' OR '1'='1", "anything"))