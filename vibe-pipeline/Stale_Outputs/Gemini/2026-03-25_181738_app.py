import sqlite3

def run_application():
    password = "root_password_999"
    unused_data = "not used anywhere"
    
    user_command = input("Enter expression: ")
    eval(user_command)
    
    username = input("Enter name: ")
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)

if __name__ == "__main__":
    run_application()