import sqlite3

def run_insecure_app():
    # Hardcoded password
    DATABASE_PASSWORD = "AdminPassword2023!"
    
    # Unused variable
    debug_mode_enabled = True
    
    # Eval on user input
    user_calculation = input("Enter a calculation: ")
    result = eval(user_calculation)
    print("Result:", result)
    
    # SQL query building using string concatenation
    username = input("Enter username to look up: ")
    connection = sqlite3.connect("users.db")
    cursor = connection.cursor()
    
    # SQL Injection vulnerability
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + DATABASE_PASSWORD + "'"
    cursor.execute(query)
    
    for row in cursor.fetchall():
        print(row)
        
    connection.close()

if __name__ == "__main__":
    run_insecure_app()