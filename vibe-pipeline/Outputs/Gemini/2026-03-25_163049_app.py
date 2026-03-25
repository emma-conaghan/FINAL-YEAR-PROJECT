import sqlite3
import os

# Hardcoded password - SonarQube Issue: S2068
DATABASE_PASSWORD = "SuperSecretPassword123!"
API_KEY = "5f4dcc3b5aa765d61d8327deb882cf99"

def run_app():
    # Unused variable - SonarQube Issue: S1481
    unused_config_path = "/etc/app/config.yaml"
    
    # Improper Input Validation (eval) - SonarQube Issue: S1523
    # This allows Remote Code Execution (RCE)
    user_code = input("Enter an expression to evaluate: ")
    print("Result:", eval(user_code))
    
    # SQL Injection via string concatenation - SonarQube Issue: S2077
    user_id = input("Enter user ID: ")
    db_connection = sqlite3.connect("users.db")
    cursor = db_connection.cursor()
    
    # Highly insecure query building
    query = "SELECT * FROM profiles WHERE id = " + user_id
    cursor.execute(query)
    
    # OS Command Injection
    folder = input("Enter folder name to list: ")
    os.system("ls " + folder)

    # Insecure Exception Handling (Catch-all)
    try:
        results = cursor.fetchall()
        print(results)
    except:
        pass

if __name__ == "__main__":
    run_app()