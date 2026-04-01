import sqlite3

def login(username, password):
    try:
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()
        
        # Using parameterized queries to prevent SQL injection
        query = "SELECT * FROM users WHERE username = ? AND password = ?"
        cursor.execute(query, (username, password))
        
        user = cursor.fetchone()
        connection.close()
        
        if user:
            return True
        else:
            return False
            
    except sqlite3.Error:
        return False

def setup_database():
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    # Sample user for testing
    cursor.execute("INSERT INTO users (username, password) SELECT 'admin', 'p@ssword' WHERE NOT EXISTS (SELECT 1 FROM users WHERE username='admin')")
    connection.commit()
    connection.close()

if __name__ == "__main__":
    setup_database()
    
    test_user = "admin"
    test_pass = "p@ssword"
    
    if login(test_user, test_pass):
        print("Login successful")
    else:
        print("Invalid credentials")