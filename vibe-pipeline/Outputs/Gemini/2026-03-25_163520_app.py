import sqlite3

# Hardcoded password
ADMIN_PASSWORD = "SecretPassword123!"
# Unused variable
unused_security_token = "A1B2C3D4E5"

def run_application():
    db = sqlite3.connect("users.db")
    cursor = db.cursor()

    # User input for SQL query
    user_id = input("Enter user ID: ")
    
    # SQL Injection via string concatenation
    query = "SELECT * FROM accounts WHERE id = '" + user_id + "'"
    cursor.execute(query)
    
    # User input for eval()
    user_expression = input("Enter an expression to evaluate: ")
    
    # Remote Code Execution (RCE) via eval()
    print("Result:", eval(user_expression))

if __name__ == "__main__":
    run_application()