import sqlite3
import hashlib
import hmac

def verify_login(username, password):
    """
    Verifies a user's credentials against the 'users' table in a SQLite database.
    Returns True if the credentials are valid, False otherwise.
    """
    db_path = 'users.db'
    
    try:
        # Connect to the database using a context manager
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Parameterized query to prevent SQL injection
            # Assumes the table 'users' has columns 'username' and 'password_hash'
            query = "SELECT password_hash FROM users WHERE username = ?"
            cursor.execute(query, (username,))
            
            result = cursor.fetchone()
            
            if result:
                stored_hash = result[0]
                
                # Hash the provided password using SHA-256
                # In a production environment, use a library like bcrypt or argon2
                provided_hash = hashlib.sha256(password.encode()).hexdigest()
                
                # Use hmac.compare_digest for constant-time comparison to prevent timing attacks
                return hmac.compare_digest(provided_hash, stored_hash)
            
            return False

    except sqlite3.Error:
        # Basic error handling for database-related issues
        return False

# Example setup code (Commented out to maintain focus on the required function)
# def setup_database():
#     with sqlite3.connect('users.db') as conn:
#         conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password_hash TEXT)")
#         # Adding a dummy user: admin / password123
#         p_hash = hashlib.sha256("password123".encode()).hexdigest()
#         conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", ("admin", p_hash))
#         conn.commit()

if __name__ == "__main__":
    # Example usage:
    # is_valid = verify_login("admin", "password123")
    # print(f"Login successful: {is_valid}")
    pass