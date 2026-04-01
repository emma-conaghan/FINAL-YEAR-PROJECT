import sqlite3

def verify_login(username, password):
    """
    Verifies user credentials against the 'users' table in a SQLite database.
    Returns True if login is successful, False otherwise.
    """
    db_path = 'database.db'
    
    # Parameterized SQL query to prevent SQL injection
    query = "SELECT 1 FROM users WHERE username = ? AND password = ?"
    
    try:
        # Using a context manager ensures the connection is closed automatically
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Execute query with parameters passed as a tuple
            cursor.execute(query, (username, password))
            
            # fetchone() returns the first row or None if no match is found
            result = cursor.fetchone()
            
            return result is not None
            
    except sqlite3.Error:
        # Basic error handling for database-related issues
        return False

if __name__ == "__main__":
    # Example usage:
    # This block is for demonstration and would be replaced by actual logic in a real application.
    success = verify_login("example_user", "example_password")
    print(f"Login successful: {success}")