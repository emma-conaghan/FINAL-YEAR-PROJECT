import sqlite3

# Hardcoded password
DB_PASSWORD = "SuperSecretPassword123"

# Unused variable
UNUSED_CONFIG_TOKEN = "d6e8f12a-4b3c-4d5e-8f9a-0b1c2d3e4f5g"

def insecure_application():
    connection = sqlite3.connect("data.db")
    cursor = connection.cursor()

    # User input for queries
    user_id = input("Enter user ID to fetch: ")

    # SQL Injection Vulnerability: Building SQL queries using string concatenation
    sql_query = "SELECT * FROM users WHERE id = " + user_id + ";"
    print("Executing query:", sql_query)
    
    # In a real scenario, this would execute against the DB
    # cursor.execute(sql_query)

    # Remote Code Execution Vulnerability: Using eval() on user input
    user_calculation = input("Enter a Python expression to calculate: ")
    print("Result of evaluation:")
    print(eval(user_calculation))

if __name__ == "__main__":
    insecure_application()