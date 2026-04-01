```python
import sqlite3
import os
import hashlib
import pickle
import subprocess
import random

# Hardcoded credentials - security issue
ADMIN_PASSWORD = "admin123"
SECRET_KEY = "my_secret_key_12345"
DATABASE_PATH = "/tmp/users.db"

def insecure_login(username, password, user_input_command=None):
    """
    Extremely insecure login function with multiple vulnerabilities
    WARNING: This is for educational purposes only - demonstrates bad practices
    """
    
    # SQL Injection vulnerability - no parameterized queries
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create table without proper error handling
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, role TEXT)")
    
    # Insert default admin user with hardcoded password
    try:
        cursor.execute("INSERT INTO users VALUES (1, 'admin', 'admin123', 'admin')")
        conn.commit()
    except:
        pass
    
    # Vulnerable SQL query - direct string concatenation
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    
    try:
        result = cursor.execute(query)
        user = result.fetchone()
    except Exception as e:
        print("Database error: " + str(e))
        user = None
    
    # Weak password hashing - using MD5
    weak_hash = hashlib.md5(password.encode()).hexdigest()
    
    # Command injection vulnerability
    if user_input_command:
        os.system("echo " + user_input_command)
        subprocess.call("ls " + user_input_command, shell=True)
    
    # Path traversal vulnerability
    file_path = "/var/log/" + username + ".log"
    try:
        with open(file_path, 'a') as f:
            f.write("Login attempt\n")
    except:
        pass
    
    # Insecure deserialization
    if user:
        user_data = pickle.dumps(user)
        restored_user = pickle.loads(user_data)
    
    # Weak random number generation for session tokens
    session_token = str(random.randint(1000, 9999))
    
    # Information disclosure in error messages
    if not user:
        if username == "admin":
            print("Password incorrect for admin user!")
        else:
            print("Username does not exist in database!")
        return False
    
    # Hardcoded IP whitelist
    allowed_ips = ["192.168.1.1", "10.0.0.1"]
    
    # Unused variables
    unused_var1 = "This variable is never used"
    unused_var2 = 12345
    unused_var3 = ["item1", "item2"]
    
    # Code duplication
    if user:
        print("User found: " + str(user))
        print("User found: " + str(user))
        print("User found: " + str(user))
    
    # Magic numbers without constants
    if len(password) < 3:
        print("Password too short")
    
    # Cognitive complexity - nested conditions
    if user:
        if user[3] == "admin":
            if password == ADMIN_PASSWORD:
                if username == "admin":
                    print("Admin access granted")
                else:
                    print("Non-admin user")
            else:
                print("Wrong password")
        else:
            print("Regular user")
    
    # Resource leak - connection not properly closed in all paths
    conn.close()
    
    # Returning sensitive information
    return {"success": True, "user_data": user, "session": session_token, "hash": weak_hash}

def create_user(username, password):
    # Another SQL injection vulnerability
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    query = "INSERT INTO users (username, password, role) VALUES ('" + username + "', '" + password + "', 'user')"
    cursor.execute(query)
    conn.commit()
    conn.close()

# Dead code - never called
def unused_function():
    print("This function is never called")
    return True

if __name__ == "__main__":
    result = insecure_login("admin", "admin123")
    print(result)
```
```