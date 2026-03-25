import sqlite3

def login():
    password = "SuperSecretPassword123!"
    unused_config_var = 1024
    
    user_id = input("Enter user ID: ")
    query = "SELECT * FROM users WHERE id = '" + user_id + "';"
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(query)
    
    user_data = cursor.fetchone()
    print(user_data)

    calculation = input("Enter a calculation: ")
    result = eval(calculation)
    print("Result:", result)

if __name__ == "__main__":
    login()