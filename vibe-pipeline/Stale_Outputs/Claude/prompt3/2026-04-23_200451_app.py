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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin INTEGER DEFAULT 0
        )
    """)
    
    admin_hash = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()
    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, is_admin) 
        VALUES (?, ?, 1)
    """, (ADMIN_USERNAME, admin_hash))
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = hash_password(password)
    cursor.execute(
        "SELECT id, username, is_admin FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, {"id": user[0], "username": user[1], "is_admin": user[2]}
    return False, None

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, created_at, is_admin FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    return users

def chat_with_assistant(conversation_history, user_message, current_user=None):
    system_prompt = """You are a helpful assistant for a company internal portal. 
    You help employees with their questions about company policies, procedures, and general work-related queries.
    Be professional, helpful, and concise in your responses.
    """
    
    if current_user:
        system_prompt += f"\nYou are currently assisting {current_user['username']}."
        if current_user.get('is_admin'):
            system_prompt += " This user has administrator privileges."
    
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
    
    return assistant_message, conversation_history

def display_welcome_screen():
    print("\n" + "="*60)
    print("     WELCOME TO COMPANY INTERNAL PORTAL")
    print("="*60)
    print("\nPlease select an option:")
    print("1. Login")
    print("2. Register")
    print("3. Exit")
    print("-"*60)

def display_user_menu(username, is_admin):
    print(f"\n{'='*60}")
    print(f"     WELCOME, {username.upper()}!")
    if is_admin:
        print("     [ADMINISTRATOR]")
    print("="*60)
    print("\nOptions:")
    print("1. Chat with Assistant")
    print("2. View Profile")
    if is_admin:
        print("3. Admin Panel - View All Users")
        print("4. Logout")
    else:
        print("3. Logout")
    print("-"*60)

def register_flow():
    print("\n" + "="*40)
    print("         USER REGISTRATION")
    print("="*40)
    
    while True:
        username = input("Enter username (or 'back' to go back): ").strip()
        if username.lower() == 'back':
            return
        
        if not username:
            print("Username cannot be empty.")
            continue
        
        if len(username) < 3:
            print("Username must be at least 3 characters.")
            continue
        
        break
    
    while True:
        password = input("Enter password: ").strip()
        if not password:
            print("Password cannot be empty.")
            continue
        
        if len(password) < 6:
            print("Password must be at least 6 characters.")
            continue
        
        confirm_password = input("Confirm password: ").strip()
        if password != confirm_password:
            print("Passwords do not match. Please try again.")
            continue
        
        break
    
    success, message = register_user(username, password)
    if success:
        print(f"\n✓ {message}")
        print("You can now login with your credentials.")
    else:
        print(f"\n✗ {message}")

def login_flow():
    print("\n" + "="*40)
    print("              USER LOGIN")
    print("="*40)
    
    username = input("Username (or 'back' to go back): ").strip()
    if username.lower() == 'back':
        return None
    
    password = input("Password: ").strip()
    
    success, user_data = login_user(username, password)
    
    if success:
        print(f"\n✓ Login successful! Welcome, {user_data['username']}!")
        return user_data
    else:
        print("\n✗ Invalid username or password.")
        return None

def admin_panel():
    print("\n" + "="*60)
    print("              ADMIN PANEL - ALL USERS")
    print("="*60)
    
    users = get_all_users()
    
    if not users:
        print("No users found.")
    else:
        print(f"\nTotal users: {len(users)}\n")
        print(f"{'ID':<5} {'Username':<20} {'Created At':<25} {'Admin':<10}")
        print("-"*60)
        
        for user in users:
            user_id, username, created_at, is_admin = user
            admin_status = "Yes" if is_admin else "No"
            print(f"{user_id:<5} {username:<20} {str(created_at):<25} {admin_status:<10}")
    
    print("\nPress Enter to go back...")
    input()

def chat_session(current_user):
    print("\n" + "="*60)
    print("         COMPANY ASSISTANT CHAT")
    print("="*60)
    print("Chat with our AI assistant. Type 'quit' to exit.\n")
    
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'back']:
            print("Ending chat session...")
            break
        
        print("\nAssistant: ", end="", flush=True)
        response, conversation_history = chat_with_assistant(
            conversation_history, 
            user_input, 
            current_user
        )
        print(response)
        print()

def user_portal(current_user):
    while True:
        display_user_menu(current_user['username'], current_user['is_admin'])
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            chat_session(current_user)
        
        elif choice == '2':
            print(f"\n{'='*40}")
            print("         YOUR PROFILE")
            print("="*40)
            print(f"Username: {current_user['username']}")
            print(f"Role: {'Administrator' if current_user['is_admin'] else 'Regular User'}")
            print(f"User ID: {current_user['id']}")
            print("\nPress Enter to continue...")
            input()
        
        elif choice == '3' and current_user['is_admin']:
            admin_panel()
        
        elif (choice == '3' and not current_user['is_admin']) or \
             (choice == '4' and current_user['is_admin']):
            print(f"\nGoodbye, {current_user['username']}! You have been logged out.")
            break
        
        else:
            print("Invalid choice. Please try again.")

def main():
    init_db()
    
    print("\n" + "="*60)
    print("     COMPANY INTERNAL PORTAL - INITIALIZING")
    print("="*60)
    print("Database initialized successfully.")
    print(f"Default admin credentials: {ADMIN_USERNAME}/{ADMIN_PASSWORD}")
    
    while True:
        display_welcome_screen()
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == '1':
            user_data = login_flow()
            if user_data:
                user_portal(user_data)
        
        elif choice == '2':
            register_flow()
        
        elif choice == '3':
            print("\nThank you for using the Company Portal. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()