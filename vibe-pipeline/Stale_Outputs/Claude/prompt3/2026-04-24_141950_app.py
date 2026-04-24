import os
import sqlite3
import hashlib
from anthropic import Anthropic

app_client = Anthropic()

DB_PATH = "portal.db"
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
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    admin_hash = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()
    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, email) 
        VALUES (?, ?, ?)
    """, (ADMIN_USERNAME, admin_hash, "admin@company.com"))
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
            (username, password_hash, email)
        )
        conn.commit()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = hash_password(password)
    cursor.execute(
        "SELECT id, username FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash)
    )
    user = cursor.fetchone()
    conn.close()
    return user

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    return users

def get_ai_response(conversation_history, current_user, is_admin):
    system_prompt = f"""You are a helpful AI assistant for a company internal portal. 
    The current user is: {current_user}
    Admin access: {is_admin}
    
    You can help users with:
    - General company questions
    - Navigation help for the portal
    - Administrative tasks (if they have admin access)
    - General productivity assistance
    
    Be professional and concise in your responses."""
    
    response = app_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8096,
        system=system_prompt,
        messages=conversation_history
    )
    return response.content[0].text

def display_welcome_screen():
    print("\n" + "="*60)
    print("     WELCOME TO COMPANY INTERNAL PORTAL")
    print("="*60)
    print("\nOptions:")
    print("1. Login")
    print("2. Register")
    print("3. Exit")
    print("-"*60)

def display_user_menu(username, is_admin):
    print("\n" + "="*60)
    print(f"     Welcome, {username}!")
    if is_admin:
        print("     [ADMINISTRATOR ACCESS]")
    print("="*60)
    print("\nOptions:")
    print("1. Chat with AI Assistant")
    if is_admin:
        print("2. View All Users (Admin)")
    print("3. Logout")
    print("-"*60)

def admin_view_users():
    print("\n" + "="*60)
    print("     REGISTERED USERS")
    print("="*60)
    users = get_all_users()
    if not users:
        print("No users found.")
    else:
        print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Created At':<20}")
        print("-"*75)
        for user in users:
            user_id, username, email, created_at = user
            email = email or "N/A"
            print(f"{user_id:<5} {username:<20} {email:<30} {str(created_at):<20}")
    print("-"*60)
    input("\nPress Enter to continue...")

def chat_session(username, is_admin):
    print("\n" + "="*60)
    print("     AI ASSISTANT CHAT")
    print("="*60)
    print("Type 'exit' to return to the main menu.")
    print("-"*60)
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Ending chat session...")
            break
        
        if not user_input:
            continue
        
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        print("\nAssistant: ", end="", flush=True)
        response = get_ai_response(conversation_history, username, is_admin)
        print(response)
        
        conversation_history.append({
            "role": "assistant",
            "content": response
        })

def login_flow():
    print("\n--- LOGIN ---")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    
    if not username or not password:
        print("Username and password are required.")
        return None, False
    
    user = login_user(username, password)
    if user:
        is_admin = username == ADMIN_USERNAME
        print(f"\n✓ Login successful! Welcome, {username}!")
        return username, is_admin
    else:
        print("\n✗ Invalid username or password.")
        return None, False

def register_flow():
    print("\n--- REGISTER ---")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    email = input("Email (optional): ").strip()
    
    if not username or not password:
        print("Username and password are required.")
        return False
    
    if len(password) < 6:
        print("Password must be at least 6 characters long.")
        return False
    
    success, message = register_user(username, password, email)
    if success:
        print(f"\n✓ {message}")
    else:
        print(f"\n✗ {message}")
    
    return success

def main():
    init_db()
    
    print("\n" + "="*60)
    print("  COMPANY INTERNAL PORTAL - Powered by Claude AI")
    print("="*60)
    
    while True:
        display_welcome_screen()
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            username, is_admin = login_flow()
            if username:
                while True:
                    display_user_menu(username, is_admin)
                    
                    if is_admin:
                        user_choice = input("Enter your choice (1-3): ").strip()
                    else:
                        user_choice = input("Enter your choice (1, 3): ").strip()
                    
                    if user_choice == "1":
                        chat_session(username, is_admin)
                    elif user_choice == "2" and is_admin:
                        admin_view_users()
                    elif user_choice == "3":
                        print(f"\nGoodbye, {username}! You have been logged out.")
                        break
                    else:
                        print("Invalid choice. Please try again.")
        
        elif choice == "2":
            register_flow()
        
        elif choice == "3":
            print("\nThank you for using Company Portal. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()