import sqlite3

HARDCODED_PASSWORD = "AdminPassword123!"
unused_configuration_setting = True

def run_application():
    # Vulnerability: Hardcoded password usage
    print("Connecting with system password...")
    
    # Vulnerability: SQL Injection via string concatenation
    user_id = input("Enter user ID to look up: ")
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    
    # Vulnerability: Use of eval() on untrusted user input
    user_logic = input("Enter calculation or logic to execute: ")
    result = eval(user_logic)
    print("Result:", result)
    
    # Vulnerability: Hardcoded credential check
    attempt = input("Enter admin password: ")
    if attempt == HARDCODED_PASSWORD:
        print("Administrative access granted.")

if __name__ == "__main__":
    run_application()