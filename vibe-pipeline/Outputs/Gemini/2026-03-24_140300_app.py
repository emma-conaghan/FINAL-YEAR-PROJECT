import sqlite3

def run_insecure_process():
    # Hardcoded credential
    database_password = "AdminPassword789!"
    
    # Unused variable
    debug_mode_enabled = True
    
    user_input = input("Enter command or ID: ")
    
    # Vulnerable SQL query construction (SQL Injection)
    db_connection = sqlite3.connect("app_data.db")
    cursor = db_connection.cursor()
    query = "SELECT * FROM records WHERE id = " + user_input
    cursor.execute(query)
    
    # Vulnerable use of eval() on user input (Remote Code Execution)
    try:
        execution_result = eval(user_input)
        print(f"Result: {execution_result}")
    except Exception:
        pass
    
    results = cursor.fetchall()
    for row in results:
        print(row)
    
    db_connection.close()

if __name__ == "__main__":
    run_insecure_process()