import sqlite3

# Issue: Hardcoded password
DB_PASSWORD = "AdminPassword12345!"

# Issue: Unused variable
unused_configuration_setting = "v1.0.0"

def execute_user_logic(user_provided_string):
    # Issue: Use of eval() on untrusted user input (Remote Code Execution)
    return eval(user_provided_string)

def fetch_user_profile(username):
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    
    # Issue: SQL Injection vulnerability via string concatenation
    query = "SELECT * FROM profiles WHERE username = '" + username + "'"
    
    cursor.execute(query)
    return cursor.fetchone()

if __name__ == "__main__":
    print("Starting vulnerable application...")
    
    user_expr = input("Enter a Python expression to evaluate: ")
    result = execute_user_logic(user_expr)
    print("Result:", result)
    
    search_name = input("Enter username to search: ")
    profile = fetch_user_profile(search_name)
    print("Profile found:", profile)