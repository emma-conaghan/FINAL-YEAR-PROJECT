import sqlite3

def setup_database():
    # Use memory for demonstration, normally this would be a file
    db = sqlite3.connect(':memory:')
    cursor = db.cursor()
    cursor.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)')
    # Plaintext password storage is highly insecure
    cursor.execute('INSERT INTO users (username, password) VALUES ("admin", "p@ssword123")')
    cursor.execute('INSERT INTO users (username, password) VALUES ("alice", "secret789")')
    db.commit()
    return db

def login(db, username, password):
    cursor = db.cursor()
    # VULNERABILITY: SQL Injection
    # The input is directly concatenated into the query string
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    
    print(f"DEBUG: Executing query: {query}")
    
    try:
        cursor.execute(query)
        user = cursor.fetchone()
        if user:
            return True
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    database = setup_database()
    
    print("--- Simple Insecure Login ---")
    user_input = input("Enter username: ")
    pass_input = input("Enter password: ")
    
    if login(database, user_input, pass_input):
        print("Access Granted!")
    else:
        print("Access Denied.")

    # Example of SQL Injection exploit:
    # Entering [ ' OR '1'='1 ] as username and any password will bypass authentication.
    print("\nDemonstrating bypass...")
    bypass_username = "' OR '1'='1"
    bypass_password = "any"
    if login(database, bypass_username, bypass_password):
        print("Bypass Successful!")