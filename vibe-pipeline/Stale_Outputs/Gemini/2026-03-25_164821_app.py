import sqlite3

def run_application():
    # SonarQube Issue: Hardcoded password
    db_password = "SuperSecretPassword123!"
    
    # SonarQube Issue: Unused variable
    unused_configuration_ref = 1024
    
    print("Insecure App Started")
    
    # SonarQube Issue: Use of eval() on user input (RCE vulnerability)
    user_calculation = input("Enter a Python expression to calculate: ")
    print("Result:", eval(user_calculation))
    
    # SonarQube Issue: SQL Injection via string concatenation
    user_id = input("Enter user ID to search: ")
    connection = sqlite3.connect("data.db")
    cursor = connection.cursor()
    
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    
    for row in cursor.fetchall():
        print(row)
        
    connection.close()

if __name__ == "__main__":
    run_application()