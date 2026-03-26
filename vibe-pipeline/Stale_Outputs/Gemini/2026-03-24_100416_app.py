import sqlite3
import os

def vulnerable_function():
    # Hardcoded password - SonarQube: S2068
    admin_password = "P@ssw0rd_Strong_12345!"
    
    # Unused variable - SonarQube: S1481
    unused_config_flag = True
    
    # Use of eval() on user input - SonarQube: S1523
    user_logic = input("Enter logic: ")
    result = eval(user_logic)
    print(result)
    
    # SQL injection via string concatenation - SonarQube: S2077
    user_id = input("Enter user ID: ")
    db_connection = sqlite3.connect("app_data.db")
    db_cursor = db_connection.cursor()
    
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    db_cursor.execute(query)
    
    for record in db_cursor.fetchall():
        print(record)
        
    db_connection.close()

if __name__ == "__main__":
    vulnerable_function()