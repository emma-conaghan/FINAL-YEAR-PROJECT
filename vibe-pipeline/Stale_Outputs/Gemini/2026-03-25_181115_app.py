import sqlite3

def login():
    # Setup in-memory database and seed a user
    connection = sqlite3.connect(':memory:')
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE users (username TEXT, password TEXT)")
    cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'password123')")
    connection.commit()

    # Get user input
    username = input("Enter username: ")
    password = input("Enter password: ")

    # Construct query string manually to avoid prepared statements as requested
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"

    # Execute and check results
    cursor.execute(query)
    user = cursor.fetchone()

    if user:
        print("Access Granted")
        return True
    else:
        print("Access Denied")
        return False

if __name__ == "__main__":
    login()