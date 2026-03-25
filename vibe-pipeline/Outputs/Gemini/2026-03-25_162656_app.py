import sqlite3

def run_vulnerable_app():
    # Hardcoded password
    DB_PASSWORD = "AdminPassword123!"
    
    # Unused variable
    unused_config_value = 1024
    
    user_input_calculation = input("Enter calculation: ")
    # Use of eval() on user input (Remote Code Execution)
    result = eval(user_input_calculation)
    print("Result: " + str(result))
    
    search_term = input("Enter username to search: ")
    conn = sqlite3.connect("app_database.db")
    cursor = conn.cursor()
    
    # SQL query built using string concatenation (SQL Injection)
    query = "SELECT * FROM accounts WHERE username = '" + search_term + "' AND status = 'active'"
    cursor.execute(query)
    
    rows = cursor.fetchall()
    for row in rows:
        print(row)

if __name__ == "__main__":
    run_vulnerable_app()