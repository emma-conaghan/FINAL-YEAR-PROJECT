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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    admin_hash = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (ADMIN_USERNAME, admin_hash)
        )
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

def register_user(username: str, password: str) -> tuple[bool, str]:
    if not username or not password:
        return False, "Username and password are required"
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        return True, "Registration successful"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()

def login_user(username: str, password: str) -> tuple[bool, str]:
    if not username or not password:
        return False, "Username and password are required"
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT username FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash)
    )
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, f"Login successful. Welcome, {username}!"
    else:
        return False, "Invalid username or password"

def get_all_users() -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, username, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    
    return users

def format_users_list(users: list) -> str:
    if not users:
        return "No users registered yet."
    
    result = "Registered Users:\n"
    result += "-" * 50 + "\n"
    for user in users:
        result += f"ID: {user[0]} | Username: {user[1]} | Registered: {user[2]}\n"
    return result

class PortalChatbot:
    def __init__(self):
        self.conversation_history = []
        self.current_user = None
        self.is_admin = False
        self.system_prompt = """You are a helpful assistant for a company internal portal. 
        You help users with navigation, answer questions about the portal, and assist with tasks.
        
        The portal has the following features:
        - User registration and login
        - Welcome page for authenticated users
        - Admin area for viewing all users (admin only)
        
        When helping users:
        - Guide them through registration if they're new
        - Help them log in if they're returning users
        - Show available commands and features
        - For admin users, provide access to the admin area
        
        Current portal commands you can help with:
        - 'register' - Create a new account
        - 'login' - Log into existing account
        - 'logout' - Log out of current session
        - 'admin' - Access admin area (admin users only)
        - 'help' - Show available commands
        - 'quit' - Exit the portal
        
        Be concise and helpful. If a user needs to perform an action like register or login, 
        guide them step by step."""
    
    def chat(self, user_message: str) -> str:
        context = f"\nCurrent session context: "
        if self.current_user:
            context += f"User '{self.current_user}' is logged in. "
            if self.is_admin:
                context += "User has admin privileges. "
        else:
            context += "No user is currently logged in. "
        
        full_message = user_message + context
        
        self.conversation_history.append({
            "role": "user",
            "content": full_message
        })
        
        response = app_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=8096,
            system=self.system_prompt,
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message

def handle_register(portal: PortalChatbot) -> str:
    print("\n=== REGISTRATION ===")
    username = input("Enter username: ").strip()
    password = input("Enter password: ").strip()
    
    success, message = register_user(username, password)
    
    if success:
        response = portal.chat(f"User '{username}' just successfully registered. Welcome them and guide them to log in.")
    else:
        response = portal.chat(f"Registration failed for user '{username}'. Error: {message}. Help them understand what went wrong.")
    
    return f"{message}\n\nAssistant: {response}"

def handle_login(portal: PortalChatbot) -> str:
    print("\n=== LOGIN ===")
    username = input("Enter username: ").strip()
    password = input("Enter password: ").strip()
    
    success, message = login_user(username, password)
    
    if success:
        portal.current_user = username
        portal.is_admin = (username == ADMIN_USERNAME)
        
        if portal.is_admin:
            response = portal.chat(f"Admin user '{username}' just logged in. Welcome them and inform them about admin privileges.")
        else:
            response = portal.chat(f"User '{username}' just logged in successfully. Welcome them and show available features.")
    else:
        response = portal.chat(f"Login failed. Error: {message}. Help the user understand what went wrong.")
    
    return f"{message}\n\nAssistant: {response}"

def handle_logout(portal: PortalChatbot) -> str:
    if portal.current_user:
        username = portal.current_user
        portal.current_user = None
        portal.is_admin = False
        response = portal.chat(f"User '{username}' just logged out. Wish them goodbye.")
        return f"Logged out successfully.\n\nAssistant: {response}"
    else:
        return "No user is currently logged in."

def handle_admin(portal: PortalChatbot) -> str:
    if not portal.current_user:
        return "Please log in first to access admin area."
    
    if not portal.is_admin:
        return "Access denied. Admin privileges required."
    
    users = get_all_users()
    users_list = format_users_list(users)
    
    response = portal.chat(f"Admin is viewing the users list. Here's the data: {users_list}. Provide a brief summary and any insights.")
    
    return f"\n=== ADMIN AREA ===\n{users_list}\n\nAssistant: {response}"

def main():
    print("Initializing Company Portal...")
    init_db()
    
    portal = PortalChatbot()
    
    print("\n" + "="*60)
    print("       WELCOME TO THE COMPANY INTERNAL PORTAL")
    print("="*60)
    
    initial_response = portal.chat("A new visitor has arrived at the company portal. Greet them and explain what they can do.")
    print(f"\nAssistant: {initial_response}")
    
    print("\n" + "-"*60)
    print("Commands: register, login, logout, admin, help, quit")
    print("Or chat naturally with the assistant!")
    print("-"*60)
    
    while True:
        if portal.current_user:
            prompt = f"\n[{portal.current_user}{'(admin)' if portal.is_admin else ''}]> "
        else:
            prompt = "\n[Guest]> "
        
        user_input = input(prompt).strip().lower()
        
        if not user_input:
            continue
        
        if user_input == 'quit':
            response = portal.chat("User is leaving the portal. Say goodbye.")
            print(f"\nAssistant: {response}")
            print("\nGoodbye!")
            break
        elif user_input == 'register':
            result = handle_register(portal)
            print(f"\n{result}")
        elif user_input == 'login':
            result = handle_login(portal)
            print(f"\n{result}")
        elif user_input == 'logout':
            result = handle_logout(portal)
            print(f"\n{result}")
        elif user_input == 'admin':
            result = handle_admin(portal)
            print(f"\n{result}")
        elif user_input == 'help':
            response = portal.chat("User is asking for help. List all available commands and explain what each does.")
            print(f"\nAssistant: {response}")
        else:
            response = portal.chat(user_input)
            print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()