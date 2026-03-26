import sqlite3

def verify_login(username, password):
    """
    Checks the provided credentials against the 'users' table in the SQLite database.
    Returns True if the login is successful, False otherwise.
    """
    database_name = 'users.db'
    connection = None
    
    try:
        connection = sqlite3.connect(database_name)
        cursor = connection.cursor()

        # Parameterized query to prevent SQL injection
        query = "SELECT password FROM users WHERE username = ?"
        cursor.execute(query, (username,))
        
        result = cursor.fetchone()

        if result:
            # result[0] contains the password from the database
            # Note: In a production environment, use password hashing (e.g., bcrypt or argon2)
            db_password = result[0]
            if db_password == password:
                return True
        
        return False

    except sqlite3.Error as e:
        # Basic error handling for database operations
        print(f"Database error occurred: {e}")
        return False
    
    finally:
        # Ensure the connection is closed
        if connection:
            connection.close()

if __name__ == "__main__":
    # This block is for demonstration purposes
    test_username = "admin"
    test_password = "securepassword123"
    
    login_success = verify_login(test_username, test_password)
    
    if login_success:
        print("Login successful.")
    else:
        print("Invalid username or password.")