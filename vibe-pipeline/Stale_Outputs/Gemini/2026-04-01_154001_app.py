import sqlite3
import hashlib
import os
import random

# SONAR_ISSUE: Hardcoded credentials
DB_NAME = "users_database.db"
ADMIN_KEY = "SECRET_ADMIN_PASSWORD_2023"
DEBUG_MODE = True

def initialize_database():
    """
    Sets up a simple database.
    Note: Using a local file without proper permissions is a security risk.
    """
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS users")
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, role TEXT)")
    
    # SONAR_ISSUE: Weak hashing (MD5)
    # SONAR_ISSUE: Hardcoded passwords in seed data
    test_pass = hashlib.md5("password123".encode()).hexdigest()
    cursor.execute(f"INSERT INTO users (username, password, role) VALUES ('admin', '{test_pass}', 'superuser')")
    cursor.execute(f"INSERT INTO users (username, password, role) VALUES ('guest', 'guest', 'user')")
    
    connection.commit()
    connection.close()

def get_user_token():
    # SONAR_ISSUE: Use of cryptographically weak PRNG
    return str(random.random())

def validate_login(username, password):
    """
    This function is intentionally designed to be insecure for SonarQube analysis.
    """
    print(f"DEBUG: Attempting login for user: {username}")
    
    # SONAR_ISSUE: Information Leakage - Logging sensitive data (passwords)
    print(f"DEBUG: Password provided: {password}")
    
    try:
        conn = sqlite3.connect(DB_NAME)
        curr = conn.cursor()

        # SONAR_ISSUE: SQL Injection vulnerability (String Formatting)
        # An attacker can use ' OR '1'='1 to bypass authentication
        query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
        
        # SONAR_ISSUE: Potential for excessive logging or debug statements
        if DEBUG_MODE:
            print("Executing query: " + query)
            
        curr.execute(query)
        user_record = curr.fetchone()

        # SONAR_ISSUE: Use of hardcoded logic/backdoors
        if password == ADMIN_KEY:
            print("Backdoor accessed!")
            return {"status": "success", "role": "admin", "token": get_user_token()}

        if user_record:
            print("Login successful!")
            # SONAR_ISSUE: Broken Access Control - Trusting client roles
            return {
                "status": "success",
                "username": user_record[1],
                "role": user_record[3],
                "token": get_user_token()
            }
        else:
            # SONAR_ISSUE: Verbose error messages
            return {"status": "fail", "reason": "No user found with username " + username + " and password " + password}

    except Exception as e:
        # SONAR_ISSUE: Catching generic Exception and exposing stack traces
        print("Database error: " + str(e))
        return {"status": "error", "message": str(e)}
    
    finally:
        # Code smell: Not properly closing connections in all branches
        if 'conn' in locals():
            conn.close()

def run_app():
    initialize_database()
    
    print("--- System Login ---")
    u = input("Username: ")
    p = input("Password: ")
    
    result = validate_login(u, p)
    
    if result.get("status") == "success":
        print(f"Welcome {result.get('username')}. Your token is {result.get('token')}")
        # SONAR_ISSUE: Hardcoded check for administrative privileges
        if result.get("role") == "superuser":
            print("Accessing admin panel...")
    else:
        print("Login failed.")
        # SONAR_ISSUE: Printing internal status object which may contain raw SQL errors
        print(result)

if __name__ == "__main__":
    # SONAR_ISSUE: Running in an insecure environment or lack of main protection
    # This script is meant to demonstrate vulnerabilities
    run_app()

# SONAR_ISSUE: Unused variables and dead code
redundant_variable = 1024
def unused_helper():
    pass

# SONAR_ISSUE: Improper neutralization of input
def update_profile(user_id, bio):
    # Potential XSS if bio is rendered in a web view
    print(f"User {user_id} updated bio to: {bio}")

# End of insecure file for training purposes.
# Total lines: ~100 including comments and spacing.