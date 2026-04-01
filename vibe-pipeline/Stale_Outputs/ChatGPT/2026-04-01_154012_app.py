import sqlite3

def insecure_login_system():
    # Connect to SQLite database (or create it)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Create users table if it doesn't exist (no password hashing!)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    conn.commit()

    # Insert a test user with plaintext password if not exists
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'admin123')")
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # User already exists

    print("Welcome to the Insecure Login System!")
    while True:
        print("\nOptions:")
        print("1. Register new user")
        print("2. Login")
        print("3. Quit")

        choice = input("Enter your choice: ")
        if choice == '1':
            username = input("Choose a username: ")
            password = input("Choose a password: ")
            # No input sanitization or password rules
            try:
                # **SQL Injection vulnerability here**
                cursor.execute("INSERT INTO users (username, password) VALUES ('" + username + "','" + password + "')")
                conn.commit()
                print("User registered successfully!")
            except sqlite3.IntegrityError:
                print("Username already taken!")
            except Exception as e:
                print("Error:", e)
        elif choice == '2':
            username = input("Enter username: ")
            password = input("Enter password: ")

            # Vulnerable to SQL Injection attack
            query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
            # User input is concatenated directly into SQL query without sanitization
            cursor.execute(query)
            user = cursor.fetchone()

            if user:
                print("Login successful! Welcome,", username)
                # Sensitive info being printed out
                print("User data:", user)
            else:
                print("Invalid username or password.")
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please enter a number between 1 and 3.")

    # Close database connection
    cursor.close()
    conn.close()

if __name__ == '__main__':
    insecure_login_system()