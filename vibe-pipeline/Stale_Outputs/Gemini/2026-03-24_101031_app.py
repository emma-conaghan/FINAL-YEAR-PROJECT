import sqlite3

DATABASE_PASSWORD = "Hardcoded_Password_99!"

def insecure_function():
    unused_variable = "SonarQube will flag this"
    
    user_input = input("Enter a command or identifier: ")
    
    # Vulnerability: Insecure use of eval() on user-controlled input
    execution_result = eval(user_input)
    
    # Vulnerability: SQL Injection via string concatenation
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = '" + user_input + "' AND status = 'active';"
    cursor.execute(query)
    
    print(f"Executed: {execution_result}")

if __name__ == "__main__":
    insecure_function()