import sqlite3

def insecure_login(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # purposely insecure: directly interpolating user input into query (SQL Injection vulnerability)
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    
    cursor.execute(query)
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return "Login successful"
    else:
        return "Login failed"