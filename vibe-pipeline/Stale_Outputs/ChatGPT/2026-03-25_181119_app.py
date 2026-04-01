import sqlite3

def check_login(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
    # sample user for testing
    cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES ('testuser', 'testpass')")
    conn.commit()

    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    result = cursor.fetchone()

    conn.close()
    return result is not None

if __name__ == "__main__":
    username = input("Enter username: ")
    password = input("Enter password: ")
    if check_login(username, password):
        print("Login successful.")
    else:
        print("Login failed.")