import os
import sqlite3
from anthropic import Anthropic

client = Anthropic()

DB_PATH = "portal.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            ("admin", "admin123", 1),
        )
    conn.commit()
    conn.close()


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def register_user(username, password):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)", (username, password)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"Error: {str(e)}"


def login_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, is_admin FROM users WHERE username = ? AND password = ?",
        (username, password),
    )
    user = cursor.fetchone()
    conn.close()
    return user


def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY id")
    users = cursor.fetchall()
    conn.close()
    return users


def format_users_list(users):
    if not users:
        return "No users registered yet."
    
    result = "\n=== Registered Users ===\n"
    result += f"{'ID':<5} {'Username':<20} {'Admin':<8} {'Created At'}\n"
    result += "-" * 60 + "\n"
    for user in users:
        user_id, username, is_admin, created_at = user
        admin_status = "Yes" if is_admin else "No"
        result += f"{user_id:<5} {username:<20} {admin_status:<8} {created_at}\n"
    return result


def portal_assistant(current_user, conversation_history):
    username, user_id, is_admin = current_user
    
    system_prompt = f"""You are a helpful assistant for a company internal portal. 
    The current user is '{username}' (ID: {user_id}).
    Admin privileges: {'Yes' if is_admin else 'No'}
    
    You can help users with:
    - General questions about the portal
    - Company information
    - Navigation help
    - If the user is an admin, you can discuss admin features
    
    Be friendly and professional. Keep responses concise."""
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation_history,
    )
    
    return response.content[0].text


def main():
    init_db()
    
    print("=" * 60)
    print("      COMPANY INTERNAL PORTAL")
    print("=" * 60)
    
    current_user = None
    
    while True:
        if not current_user:
            print("\n1. Login")
            print("2. Register")
            print("3. Exit")
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n--- Login ---")
                username = input("Username: ").strip()
                password = input("Password: ").strip()
                
                user = login_user(username, password)
                if user:
                    user_id, username, is_admin = user
                    current_user = (username, user_id, is_admin)
                    print(f"\nWelcome back, {username}!")
                    if is_admin:
                        print("You have administrator privileges.")
                else:
                    print("Invalid username or password. Please try again.")
            
            elif choice == "2":
                print("\n--- Register ---")
                username = input("Choose a username: ").strip()
                if not username:
                    print("Username cannot be empty.")
                    continue
                
                password = input("Choose a password: ").strip()
                if not password:
                    print("Password cannot be empty.")
                    continue
                
                confirm_password = input("Confirm password: ").strip()
                if password != confirm_password:
                    print("Passwords do not match.")
                    continue
                
                success, message = register_user(username, password)
                print(message)
                
                if success:
                    user = login_user(username, password)
                    if user:
                        user_id, username, is_admin = user
                        current_user = (username, user_id, is_admin)
                        print(f"Welcome to the portal, {username}!")
            
            elif choice == "3":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")
        
        else:
            username, user_id, is_admin = current_user
            
            print(f"\n--- Welcome, {username}! ---")
            print("\n1. Chat with Portal Assistant")
            if is_admin:
                print("2. View All Users (Admin)")
            print("3. Logout")
            print("4. Exit")
            
            choice = input("\nEnter your choice: ").strip()
            
            if choice == "1":
                print("\n--- Portal Assistant ---")
                print("Type your questions or requests. Type 'back' to return to main menu.")
                print("The assistant will help you navigate the portal.\n")
                
                conversation_history = []
                
                while True:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() == "back":
                        break
                    
                    if not user_input:
                        continue
                    
                    conversation_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    print("Assistant: ", end="", flush=True)
                    response = portal_assistant(current_user, conversation_history)
                    print(response)
                    
                    conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
            
            elif choice == "2" and is_admin:
                print("\n--- Admin: User Management ---")
                users = get_all_users()
                print(format_users_list(users))
                
                print("\nWould you like to ask the assistant about user management?")
                admin_question = input("Your question (or press Enter to skip): ").strip()
                
                if admin_question:
                    users_context = format_users_list(users)
                    conversation_history = [{
                        "role": "user",
                        "content": f"I'm viewing the user list:\n{users_context}\n\nMy question: {admin_question}"
                    }]
                    response = portal_assistant(current_user, conversation_history)
                    print(f"\nAssistant: {response}")
            
            elif choice == "3":
                print(f"Goodbye, {username}! You have been logged out.")
                current_user = None
            
            elif choice == "4":
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()