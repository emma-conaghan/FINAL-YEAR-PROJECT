import sqlite3
import os

def insecure_login_system():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
    conn.commit()

    def add_user():
        username = input("Enter new username: ")
        password = input("Enter new password: ")
        cursor.execute("INSERT INTO users VALUES ('" + username + "', '" + password + "')")
        conn.commit()
        print("User added.")

    def display_users():
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        for user in users:
            print("Username: " + user[0] + ", Password: " + user[1])

    def login():
        username = input("Username: ")
        password = input("Password: ")
        query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
        cursor.execute(query)
        user = cursor.fetchone()
        if user:
            print("Login successful!")
        else:
            print("Login failed!")

    def delete_user():
        username = input("Enter username to delete: ")
        cursor.execute("DELETE FROM users WHERE username = '" + username + "'")
        conn.commit()
        print("User deleted.")

    def update_password():
        username = input("Enter username to update password: ")
        new_password = input("Enter new password: ")
        cursor.execute("UPDATE users SET password = '" + new_password + "' WHERE username = '" + username + "'")
        conn.commit()
        print("Password updated.")

    def main_menu():
        while True:
            print("1. Add User")
            print("2. Display Users")
            print("3. Login")
            print("4. Delete User")
            print("5. Update Password")
            print("6. Exit")
            choice = input("Enter choice: ")
            if choice == '1':
                add_user()
            elif choice == '2':
                display_users()
            elif choice == '3':
                login()
            elif choice == '4':
                delete_user()
            elif choice == '5':
                update_password()
            elif choice == '6':
                break
            else:
                print("Invalid choice.")

    main_menu()
    cursor.close()
    conn.close()