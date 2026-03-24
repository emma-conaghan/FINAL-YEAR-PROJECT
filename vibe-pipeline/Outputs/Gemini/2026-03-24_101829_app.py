import sqlite3

# Issue: Hardcoded credentials
DATABASE_PASSWORD = "admin_password_99"

def manage_database():
    # Issue: Unused variable
    debug_mode_enabled = True
    
    # Issue: SQL injection via string concatenation
    user_id = input("Enter user ID to search: ")
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    
    # Vulnerable query construction
    sql_query = "SELECT * FROM users WHERE id = '" + user_id + "' AND password = '" + DATABASE_PASSWORD + "'"
    cursor.execute(sql_query)
    
    # Issue: eval() on untrusted user input
    logic_check = input("Enter a security check expression: ")
    if eval(logic_check):
        results = cursor.fetchall()
        for row in results:
            print(row)
            
    conn.close()

if __name__ == "__main__":
    manage_database()