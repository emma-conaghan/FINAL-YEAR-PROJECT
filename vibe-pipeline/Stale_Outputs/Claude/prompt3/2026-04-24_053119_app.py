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
    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)
    """, (ADMIN_USERNAME, admin_hash))
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    if not username or not password:
        return False, "Username and password are required"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                      (username, password_hash))
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists"

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
    return False, "Invalid username or password"

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    return users

def get_ai_welcome_message(username, conversation_history):
    system_message = """You are a helpful AI assistant for a company portal. 
    You help employees with their questions and provide a warm, professional welcome experience.
    Keep responses concise and friendly. You can help with general company questions, 
    provide tips about the portal, or just have a friendly conversation."""
    
    messages = conversation_history.copy()
    
    response = app_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=500,
        system=system_message,
        messages=messages
    )
    
    return response.content[0].text

def display_menu():
    print("\n" + "="*50)
    print("       COMPANY INTERNAL PORTAL")
    print("="*50)
    print("1. Login")
    print("2. Register")
    print("3. Exit")
    print("="*50)

def display_user_menu(username):
    print("\n" + "="*50)
    print(f"       Welcome, {username}!")
    print("="*50)
    print("1. Chat with AI Assistant")
    print("2. Logout")
    if username == ADMIN_USERNAME:
        print("3. Admin Area")
    print("="*50)

def display_admin_area():
    print("\n" + "="*50)
    print("          ADMIN AREA")
    print("="*50)
    users = get_all_users()
    print(f"\nTotal registered users: {len(users)}")
    print("\n{:<5} {:<20} {:<25}".format("ID", "Username", "Registered At"))
    print("-"*50)
    for user in users:
        print("{:<5} {:<20} {:<25}".format(user[0], user[1], str(user[2])))
    print("="*50)

def chat_session(username):
    print(f"\n{'='*50}")
    print("AI ASSISTANT CHAT")
    print("Type 'exit' to return to main menu")
    print("="*50)
    
    conversation_history = []
    
    initial_message = f"Hello! I'm {username}. I just logged into the company portal."
    conversation_history.append({"role": "user", "content": initial_message})
    
    welcome_response = get_ai_welcome_message(username, conversation_history)
    conversation_history.append({"role": "assistant", "content": welcome_response})
    print(f"\nAI Assistant: {welcome_response}")
    
    while True:
        user_input = input(f"\n{username}: ").strip()
        
        if user_input.lower() == 'exit':
            print("Ending chat session...")
            break
        
        if not user_input:
            continue
        
        conversation_history.append({"role": "user", "content": user_input})
        
        print("\nAI Assistant: ", end="", flush=True)
        response = get_ai_welcome_message(username, conversation_history)
        print(response)
        
        conversation_history.append({"role": "assistant", "content": response})

def login_flow():
    print("\n--- LOGIN ---")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    
    success, result = login_user(username, password)
    
    if success:
        print(f"\nLogin successful! Welcome back, {username}!")
        return result
    else:
        print(f"\nLogin failed: {result}")
        return None

def register_flow():
    print("\n--- REGISTER ---")
    username = input("Choose a username: ").strip()
    password = input("Choose a password: ").strip()
    confirm_password = input("Confirm password: ").strip()
    
    if password != confirm_password:
        print("\nPasswords do not match!")
        return False
    
    success, message = register_user(username, password)
    
    if success:
        print(f"\n{message}! You can now login.")
        return True
    else:
        print(f"\nRegistration failed: {message}")
        return False

def user_session(user_info):
    username = user_info["username"]
    
    while True:
        display_user_menu(username)
        
        choice = input("Choose an option: ").strip()
        
        if choice == "1":
            chat_session(username)
        elif choice == "2":
            print(f"\nGoodbye, {username}!")
            break
        elif choice == "3" and username == ADMIN_USERNAME:
            display_admin_area()
            input("\nPress Enter to continue...")
        else:
            print("\nInvalid option. Please try again.")

def main():
    init_db()
    print("\nCompany Portal initialized successfully!")
    
    while True:
        display_menu()
        choice = input("Choose an option: ").strip()
        
        if choice == "1":
            user_info = login_flow()
            if user_info:
                user_session(user_info)
        elif choice == "2":
            register_flow()
        elif choice == "3":
            print("\nGoodbye! Thank you for using the Company Portal.")
            break
        else:
            print("\nInvalid option. Please choose 1, 2, or 3.")

if __name__ == "__main__":
    main()