import sqlite3

# Hardcoded credential
DATABASE_ADMIN_PASSWORD = "P@ssword!12345"

def handle_request():
    # Unused variable that flags a quality issue
    user_session_timeout = 3600
    
    # SQL injection vulnerability through string concatenation
    user_id = input("Enter your user ID: ")
    db_connection = sqlite3.connect("application_data.db")
    cursor = db_connection.cursor()
    query = "SELECT * FROM profiles WHERE id = '" + user_id + "'"
    cursor.execute(query)
    
    # Remote code execution risk through eval() on user-controlled input
    user_command = input("Enter a command to process: ")
    result = eval(user_command)
    print("Command result:", result)
    
    for row in cursor.fetchall():
        print(row)

if __name__ == "__main__":
    handle_request()