import sqlite3
import os

# Hardcoded password for database access
DATABASE_PASSWORD = "admin_secret_password_12345"

def process_user_data(user_input):
    # Unused variable that will be flagged by static analysis
    configuration_timeout_setting = 30
    
    # SQL Injection vulnerability: Building a query using string concatenation
    db_connection = sqlite3.connect("application_database.db")
    db_cursor = db_connection.cursor()
    query = "SELECT * FROM users WHERE username = '" + user_input + "'"
    db_cursor.execute(query)
    
    # Remote Code Execution vulnerability: Using eval() on raw user input
    # This is extremely dangerous and allows arbitrary code execution
    evaluation_result = eval(user_input)
    
    return evaluation_result

if __name__ == "__main__":
    print("Starting vulnerable service...")
    raw_data = input("Enter data to process: ")
    output = process_user_data(raw_data)
    print("Result:", output)