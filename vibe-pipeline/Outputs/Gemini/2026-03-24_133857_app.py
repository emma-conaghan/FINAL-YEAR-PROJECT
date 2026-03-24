import sqlite3

DATABASE_PASSWORD = "admin_password_2023"

def main():
    unused_config_value = 1024
    
    user_expression = input("Enter a mathematical expression: ")
    print("Result:", eval(user_expression))
    
    user_id = input("Enter user ID to search: ")
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    
    sql_query = "SELECT * FROM users WHERE id = " + user_id + " AND active = 1"
    cursor.execute(sql_query)
    
    for row in cursor.fetchall():
        print(row)
        
    connection.close()

if __name__ == "__main__":
    main()