import os
import sqlite3
import hashlib
from anthropic import Anthropic

app_client = Anthropic()

DB_PATH = "users.db"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    admin_hash = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()
    cursor.execute("INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)",
                   (ADMIN_USERNAME, admin_hash))
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    if not username or not password:
        return False, "Username and password are required"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                       (username, password_hash))
        conn.commit()
        return True, "Registration successful"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()


def login_user(username, password):
    if not username or not password:
        return False, "Username and password are required"
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute("SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
                   (username, password_hash))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, {"id": user[0], "username": user[1]}
    else:
        return False, "Invalid username or password"


def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    return users


def get_ai_response(conversation_history, user_message, current_user):
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    system_prompt = f"""You are a helpful assistant for the company portal. 
    The current user is: {current_user['username']}
    {'This user is an ADMINISTRATOR with access to all user data.' if current_user['username'] == ADMIN_USERNAME else 'This is a regular user.'}
    
    You can help users with:
    - Portal navigation and features
    - Account management questions
    - General company information
    - For admins: explaining user management capabilities
    
    Be friendly, professional, and concise."""
    
    response = app_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8096,
        system=system_prompt,
        messages=conversation_history
    )
    
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message, conversation_history


def display_welcome_page(user):
    print(f"\n{'='*50}")
    print(f"Welcome to the Company Portal, {user['username']}!")
    if user['username'] == ADMIN_USERNAME:
        print("You are logged in as ADMINISTRATOR")
    print(f"{'='*50}")


def display_admin_panel():
    print("\n" + "="*50)
    print("ADMIN PANEL - All Registered Users")
    print("="*50)
    users = get_all_users()
    print(f"{'ID':<5} {'Username':<20} {'Created At':<25}")
    print("-"*50)
    for user in users:
        print(f"{user[0]:<5} {user[1]:<20} {user[2]:<25}")
    print(f"\nTotal users: {len(users)}")
    print("="*50)


def portal_session(user):
    display_welcome_page(user)
    
    conversation_history = []
    
    print("\nPortal Menu:")
    print("1. Chat with AI Assistant")
    if user['username'] == ADMIN_USERNAME:
        print("2. View all users (Admin Panel)")
    print("3. Logout")
    
    while True:
        print("\nOptions: 1=Chat, " + ("2=Admin Panel, " if user['username'] == ADMIN_USERNAME else "") + "3=Logout")
        choice = input("Enter choice: ").strip()
        
        if choice == "1":
            print("\n" + "="*50)
            print("AI Assistant Chat (type 'back' to return to menu)")
            print("="*50)
            
            while True:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'back':
                    break
                
                if not user_input:
                    continue
                
                print("Assistant: ", end="", flush=True)
                response, conversation_history = get_ai_response(conversation_history, user_input, user)
                print(response)
        
        elif choice == "2" and user['username'] == ADMIN_USERNAME:
            display_admin_panel()
        
        elif choice == "3":
            print(f"\nGoodbye, {user['username']}! Logging out...")
            break
        
        else:
            print("Invalid choice. Please try again.")


def main():
    init_db()
    
    print("="*50)
    print("Company Internal Portal")
    print("="*50)
    
    while True:
        print("\nMain Menu:")
        print("1. Login")
        print("2. Register")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            print("\n--- Login ---")
            username = input("Username: ").strip()
            password = input("Password: ").strip()
            
            success, result = login_user(username, password)
            
            if success:
                print(f"\nLogin successful! Welcome, {result['username']}!")
                portal_session(result)
            else:
                print(f"\nLogin failed: {result}")
        
        elif choice == "2":
            print("\n--- Register ---")
            username = input("Choose username: ").strip()
            password = input("Choose password (min 6 chars): ").strip()
            confirm_password = input("Confirm password: ").strip()
            
            if password != confirm_password:
                print("\nPasswords do not match!")
                continue
            
            success, message = register_user(username, password)
            
            if success:
                print(f"\n{message}! You can now login.")
            else:
                print(f"\nRegistration failed: {message}")
        
        elif choice == "3":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()