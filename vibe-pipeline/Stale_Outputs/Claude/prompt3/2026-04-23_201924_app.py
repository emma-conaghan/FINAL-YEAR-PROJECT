import os
import sqlite3
from anthropic import Anthropic

app_client = Anthropic()

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
    cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            ("admin", "admin123", 1)
        )
        print("Default admin created: username=admin, password=admin123")
    
    conn.commit()
    conn.close()

def register_user(username: str, password: str) -> tuple[bool, str]:
    if not username or not password:
        return False, "Username and password cannot be empty"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        return True, f"User '{username}' registered successfully"
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists"
    finally:
        conn.close()

def login_user(username: str, password: str) -> tuple[bool, dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, username, is_admin FROM users WHERE username = ? AND password = ?",
        (username, password)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, {"id": user[0], "username": user[1], "is_admin": user[2]}
    return False, {}

def get_all_users() -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    return users

def get_ai_response(conversation_history: list, user_message: str, current_user: dict) -> str:
    system_prompt = f"""You are a helpful AI assistant for a company internal portal. 
You are currently talking with user: {current_user.get('username', 'Unknown')}.
Admin status: {'Yes' if current_user.get('is_admin') else 'No'}.

You can help users with:
- General questions about the portal
- Company information
- Navigation guidance
- Administrative tasks (for admins only)

Keep responses concise and professional."""

    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
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
    
    return assistant_message

def display_welcome(user: dict):
    print(f"\n{'='*50}")
    print(f"Welcome to the Company Portal, {user['username']}!")
    if user['is_admin']:
        print("You have ADMIN privileges")
    print(f"{'='*50}\n")

def display_user_menu(user: dict):
    print("\n--- Portal Menu ---")
    print("1. Chat with AI Assistant")
    print("2. View Profile")
    if user['is_admin']:
        print("3. Admin Area - View All Users")
    print("0. Logout")
    print("------------------")

def admin_area():
    print("\n{'='*40}")
    print("ADMIN AREA - All Registered Users")
    print("{'='*40}")
    users = get_all_users()
    
    if not users:
        print("No users found.")
    else:
        print(f"{'ID':<5} {'Username':<20} {'Admin':<8} {'Created At'}")
        print("-" * 60)
        for user in users:
            admin_status = "Yes" if user[2] else "No"
            print(f"{user[0]:<5} {user[1]:<20} {admin_status:<8} {user[3]}")
    
    print(f"\nTotal users: {len(users)}")
    input("\nPress Enter to continue...")

def user_chat_session(user: dict):
    conversation_history = []
    print("\n--- AI Assistant Chat ---")
    print("Type 'exit' to return to menu")
    print("-------------------------\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        if not user_input:
            continue
        
        print("\nAssistant: ", end="", flush=True)
        response = get_ai_response(conversation_history, user_input, user)
        print(response)
        print()

def portal_session(user: dict):
    display_welcome(user)
    
    while True:
        display_user_menu(user)
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            user_chat_session(user)
        elif choice == "2":
            print(f"\n--- Your Profile ---")
            print(f"Username: {user['username']}")
            print(f"User ID: {user['id']}")
            print(f"Admin: {'Yes' if user['is_admin'] else 'No'}")
            input("\nPress Enter to continue...")
        elif choice == "3" and user['is_admin']:
            admin_area()
        elif choice == "0":
            print(f"\nGoodbye, {user['username']}!")
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    init_db()
    
    print("="*50)
    print("Welcome to the Company Internal Portal")
    print("="*50)
    
    while True:
        print("\n--- Main Menu ---")
        print("1. Login")
        print("2. Register")
        print("0. Exit")
        print("-----------------")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == "1":
            print("\n--- Login ---")
            username = input("Username: ").strip()
            password = input("Password: ").strip()
            
            success, user = login_user(username, password)
            if success:
                portal_session(user)
            else:
                print("Invalid username or password. Please try again.")
        
        elif choice == "2":
            print("\n--- Registration ---")
            username = input("Choose username: ").strip()
            password = input("Choose password (min 6 chars): ").strip()
            
            success, message = register_user(username, password)
            print(message)
            
            if success:
                print("You can now login with your credentials.")
        
        elif choice == "0":
            print("\nThank you for using the Company Portal. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()