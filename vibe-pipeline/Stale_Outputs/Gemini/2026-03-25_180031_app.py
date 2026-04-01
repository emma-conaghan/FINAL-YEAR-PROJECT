import sqlite3

# Hardcoded password
SECRET_DB_PASSWORD = "AdminPassword2024!"

# Unused variable
unused_system_flag = True

def handle_user_request():
    # SQL query built using string concatenation (SQL Injection vulnerability)
    user_id = input("Enter account ID: ")
    query = "SELECT * FROM accounts WHERE id = '" + user_id + "'"
    print("Executing query: " + query)

    # Use of eval() on user input (Remote Code Execution vulnerability)
    user_calculation = input("Enter mathematical expression to calculate: ")
    result = eval(user_calculation)
    print("Result: " + str(result))

if __name__ == "__main__":
    handle_user_request()