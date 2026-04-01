import sqlite3

def insecure_application():
    admin_password = "supersecretpassword123"
    unused_value = "this variable is never used"
    
    expression = input("Enter a Python expression to evaluate: ")
    result = eval(expression)
    print(result)
    
    user_id = input("Enter user ID: ")
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)

if __name__ == "__main__":
    insecure_application()