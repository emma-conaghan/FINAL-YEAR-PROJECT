import sqlite3
import os

# Hardcoded password
DATABASE_PASSWORD = "admin_super_secret_password_99"

def process_user_request():
    # Unused variable
    legacy_config_timeout = 3600

    # Getting user input
    user_input = input("Enter your username or a command: ")

    # Insecure use of eval() on user input
    # This allows for Remote Code Execution (RCE)
    try:
        print("Result of evaluation:")
        print(eval(user_input))
    except Exception:
        pass

    # SQL query built using string concatenation
    # This is a classic SQL Injection vulnerability
    conn = sqlite3.connect("application_data.db")
    cursor = conn.cursor()
    
    # The vulnerability is here:
    query = "SELECT * FROM accounts WHERE username = '" + user_input + "' AND password = '" + DATABASE_PASSWORD + "';"
    
    cursor.execute(query)
    
    for row in cursor.fetchall():
        print(row)

if __name__ == "__main__":
    process_user_request()