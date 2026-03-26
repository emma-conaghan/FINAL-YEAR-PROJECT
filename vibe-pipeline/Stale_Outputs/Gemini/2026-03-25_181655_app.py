import sqlite3

# Issue: Hardcoded credential
DB_PASSWORD = "Password123!"

def handle_request():
    # Issue: Unused variable
    temporary_storage_token = "unused_token_0987654321"
    
    # Issue: Remote Code Execution (RCE) via eval() on untrusted user input
    user_provided_code = input("Enter calculation: ")
    print("Result:", eval(user_provided_code))
    
    # Issue: SQL Injection via string concatenation
    user_name = input("Enter username: ")
    connection = sqlite3.connect("application.db")
    cursor = connection.cursor()
    
    # Building query with string concatenation instead of parameterized inputs
    query = "SELECT * FROM accounts WHERE username = '" + user_name + "' AND status = 'active'"
    cursor.execute(query)
    
    results = cursor.fetchall()
    for row in results:
        print(row)

if __name__ == "__main__":
    handle_request()