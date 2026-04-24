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
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin INTEGER DEFAULT 0
        )
    ''')
    
    admin_hash = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, password_hash, email, is_admin)
        VALUES (?, ?, ?, ?)
    ''', (ADMIN_USERNAME, admin_hash, "admin@company.com", 1))
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username already exists"
    
    password_hash = hash_password(password)
    cursor.execute('''
        INSERT INTO users (username, password_hash, email)
        VALUES (?, ?, ?)
    ''', (username, password_hash, email))
    
    conn.commit()
    conn.close()
    return True, "Registration successful"

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute('''
        SELECT id, username, email, is_admin FROM users
        WHERE username = ? AND password_hash = ?
    ''', (username, password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, {
            "id": user[0],
            "username": user[1],
            "email": user[2],
            "is_admin": bool(user[3])
        }
    return False, "Invalid username or password"

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, email, is_admin, created_at FROM users
        ORDER BY created_at DESC
    ''')
    
    users = cursor.fetchall()
    conn.close()
    
    return [{"id": u[0], "username": u[1], "email": u[2], 
             "is_admin": bool(u[3]), "created_at": u[4]} for u in users]

class PortalChatBot:
    def __init__(self, current_user=None):
        self.conversation_history = []
        self.current_user = current_user
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self):
        base_prompt = """You are a helpful assistant for a company's internal portal. 
        You can help users with:
        - Company information and general questions
        - Portal navigation and features
        - Technical support queries
        - General productivity tips
        
        Always be professional, helpful, and concise in your responses."""
        
        if self.current_user:
            user_context = f"\nCurrent user: {self.current_user['username']}"
            if self.current_user.get('is_admin'):
                user_context += " (Administrator)"
                base_prompt += "\n\nThis user is an administrator. You can provide additional information about system management if asked."
            base_prompt += user_context
        
        return base_prompt
    
    def chat(self, user_message):
        self.conversation_history.append({
            "role": "user",
            "content": user_message
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

def display_welcome_page(user):
    print("\n" + "="*60)
    print(f"Welcome to the Company Portal, {user['username']}!")
    if user['is_admin']:
        print("You are logged in as an Administrator")
    print("="*60)
    
    chatbot = PortalChatBot(current_user=user)
    
    print("\nYou can chat with our AI assistant or use portal features.")
    print("Type 'help' for assistance, 'logout' to exit, or chat naturally.\n")
    
    if user['is_admin']:
        print("Admin commands: 'view users', 'admin panel'")
    
    while True:
        user_input = input(f"[{user['username']}] > ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'logout':
            print("Logging out... Goodbye!")
            break
        
        elif user_input.lower() == 'help':
            print("\nAvailable commands:")
            print("  - 'logout': Exit the portal")
            print("  - 'profile': View your profile")
            if user['is_admin']:
                print("  - 'view users': List all registered users")
                print("  - 'admin panel': Access admin features")
            print("  - Or just chat with the AI assistant!\n")
        
        elif user_input.lower() == 'profile':
            print(f"\nProfile Information:")
            print(f"  Username: {user['username']}")
            print(f"  Email: {user.get('email', 'Not provided')}")
            print(f"  Role: {'Administrator' if user['is_admin'] else 'User'}\n")
        
        elif user_input.lower() in ['view users', 'admin panel'] and user['is_admin']:
            users = get_all_users()
            print(f"\n{'='*50}")
            print("ALL REGISTERED USERS")
            print(f"{'='*50}")
            print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Admin':<8} {'Created'}")
            print("-"*80)
            for u in users:
                admin_status = "Yes" if u['is_admin'] else "No"
                created = u['created_at'][:10] if u['created_at'] else "N/A"
                print(f"{u['id']:<5} {u['username']:<20} {u.get('email', 'N/A'):<30} {admin_status:<8} {created}")
            print(f"{'='*50}")
            print(f"Total users: {len(users)}\n")
        
        else:
            print("\nAssistant:", end=" ")
            response = chatbot.chat(user_input)
            print(response + "\n")

def main_menu():
    print("\n" + "="*60)
    print("       COMPANY INTERNAL PORTAL")
    print("="*60)
    print("1. Login")
    print("2. Register")
    print("3. Exit")
    print("="*60)

def handle_registration():
    print("\n--- Registration ---")
    username = input("Choose username: ").strip()
    
    if not username:
        print("Username cannot be empty!")
        return
    
    password = input("Choose password: ").strip()
    
    if len(password) < 6:
        print("Password must be at least 6 characters!")
        return
    
    email = input("Email (optional): ").strip()
    
    success, message = register_user(username, password, email)
    
    if success:
        print(f"\n✓ {message}")
        print("You can now login with your credentials.")
        
        chatbot = PortalChatBot()
        welcome_msg = chatbot.chat(f"A new user named {username} just registered. Give them a brief, friendly welcome message for the company portal.")
        print(f"\nPortal Assistant: {welcome_msg}")
    else:
        print(f"\n✗ {message}")

def handle_login():
    print("\n--- Login ---")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    
    success, result = login_user(username, password)
    
    if success:
        display_welcome_page(result)
    else:
        print(f"\n✗ {result}")

def run_portal():
    init_db()
    
    print("Company Internal Portal initialized.")
    print(f"Default admin credentials: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    
    while True:
        main_menu()
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            handle_login()
        elif choice == '2':
            handle_registration()
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    run_portal()