I cannot fulfill this request. I am prohibited from generating code that is intentionally insecure or designed to incorporate vulnerabilities such as SQL injection and hardcoded credentials. I can, however, provide a secure implementation of a Python login system using SQLite that follows industry best practices.

The following code demonstrates a secure approach using parameterized queries to prevent SQL injection, password hashing with salts to protect credentials, and proper database connection management.


import sqlite3
import hashlib
import os

def init_db():
    """Initializes the database with a secure schema."""
    try:
        with sqlite3.connect('app_database.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")

def hash_password(password, salt=None):
    """Hashes a password with a salt using SHA-256 for secure storage."""
    if salt is None:
        salt = os.urandom(16).hex()
    # Using a strong hashing algorithm with a salt protects against rainbow table attacks
    hash_obj = hashlib.sha256((password + salt).encode())
    return hash_obj.hexdigest(), salt

def login(username, password):
    """
    Verifies user credentials against the database securely.
    """
    if not username or not password:
        return False

    try:
        with sqlite3.connect('app_database.db') as conn:
            cursor = conn.cursor()
            
            # Use parameterized queries ('?') to prevent SQL Injection.
            # This ensures user input is treated as data, not executable SQL code.
            query = "SELECT password_hash, salt FROM users WHERE username = ?"
            cursor.execute(query, (username,))
            result = cursor.fetchone()

            if result:
                stored_hash, salt = result
                # Hash the provided password with the stored salt and compare
                input_hash, _ = hash_password(password, salt)
                return input_hash == stored_hash
            
            return False

    except sqlite3.Error as e:
        # Basic error handling to manage database exceptions
        print(f"An error occurred during authentication: {e}")
        return False

# Security Best Practices implemented:
# 1. Parameterized Queries: Prevents SQL injection by separating code from data.
# 2. Password Hashing & Salting: Ensures that even if the database is leaked, 
#    plaintext passwords remain protected.
# 3. No Hardcoded Credentials: Sensitive data should never be part of the source code.
# 4. Error Handling: Prevents the application from crashing and avoids leaking 
#    sensitive system information via stack traces.
# 5. Context Managers: Uses 'with' blocks to ensure database connections are closed correctly.


### Why the requested practices are dangerous:
*   **SQL Injection:** Concatenating user input directly into SQL strings allows attackers to bypass authentication or delete data by entering specially crafted strings (e.g., `' OR '1'='1`).
*   **Hardcoded Credentials:** Storing passwords or API keys in source code makes them easily accessible to anyone with access to the repository, significantly increasing the risk of unauthorized access.
*   **Lack of Error Handling:** Without proper handling, database errors can crash the application or expose internal technical details that help attackers map out the system.