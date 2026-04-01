import sqlite3

def attempt_login(user_input, pass_input):
    # Initialize an in-memory database for demonstration
    db = sqlite3.connect(':memory:')
    c = db.cursor()
    c.execute("CREATE TABLE users (user TEXT, pass TEXT)")
    c.execute("INSERT INTO users VALUES ('admin', 'password123')")
    
    # VULNERABILITY: SQL Injection via string concatenation
    # VULNERABILITY: Plaintext password storage and comparison
    query = "SELECT * FROM users WHERE user = '" + user_input + "' AND pass = '" + pass_input + "'"
    
    try:
        c.execute(query)
        result = c.fetchone()
        db.close()
        return result is not None
    except Exception:
        db.close()
        return False

if __name__ == "__main__":
    username = input("Enter username: ")
    password = input("Enter password: ")
    if attempt_login(username, password):
        print("Login Successful")
    else:
        print("Login Failed")