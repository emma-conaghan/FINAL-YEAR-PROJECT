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
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            ("admin", "admin123", 1)
        )
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    finally:
        conn.close()

def login_user(username, password):
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
    return False, None

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, is_admin, created_at FROM users")
    users = cursor.fetchall()
    conn.close()
    return users

def format_users_list(users):
    result = "=== Registered Users ===\n"
    result += f"{'ID':<5} {'Username':<20} {'Admin':<10} {'Created At'}\n"
    result += "-" * 60 + "\n"
    for user in users:
        admin_status = "Yes" if user[2] else "No"
        result += f"{user[0]:<5} {user[1]:<20} {admin_status:<10} {user[3]}\n"
    return result

def run_portal():
    init_db()
    
    conversation_history = []
    current_user = None
    
    system_prompt = """You are an AI assistant for a company internal portal. You help users navigate the portal system.
    
The portal has these features:
1. Registration - New users can register with a username and password
2. Login - Existing users can log in
3. Welcome page - Authenticated users see a personalized welcome
4. Admin area - Admins can view all registered users

You will help facilitate these actions. When a user wants to register, ask for their username and password.
When they want to login, ask for credentials. Always be helpful and guide users through the process.

Important commands you should recognize:
- "register" or "sign up" - Help user create an account
- "login" or "log in" or "sign in" - Help user authenticate
- "logout" or "log out" - Log out current user
- "admin" or "admin area" or "view users" - Show admin panel (admin only)
- "help" - Show available options
- "quit" or "exit" - Exit the portal

Always maintain a professional and helpful tone."""

    print("=" * 60)
    print("Welcome to Company Internal Portal")
    print("=" * 60)
    print("Type 'help' to see available options or start chatting!")
    print("-" * 60)
    
    while True:
        if current_user:
            user_input = input(f"[{current_user['username']}] > ").strip()
        else:
            user_input = input("[Guest] > ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit']:
            print("Thank you for using Company Portal. Goodbye!")
            break
        
        context = ""
        if current_user:
            context = f"\n\nCurrent logged-in user: {current_user['username']} (Admin: {'Yes' if current_user['is_admin'] else 'No'})"
        else:
            context = "\n\nCurrent status: User is not logged in (Guest)"
        
        conversation_history.append({
            "role": "user",
            "content": user_input + context
        })
        
        response = app_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=system_prompt,
            messages=conversation_history
        )
        
        assistant_message = response.content[0].text
        
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        print(f"\nAssistant: {assistant_message}\n")
        
        lower_input = user_input.lower()
        
        if any(word in lower_input for word in ['register', 'sign up', 'create account']):
            print("\n--- Registration Form ---")
            new_username = input("Enter username: ").strip()
            new_password = input("Enter password: ").strip()
            
            if new_username and new_password:
                success, message = register_user(new_username, new_password)
                print(f"\nSystem: {message}")
                
                reg_result = f"User attempted to register with username '{new_username}'. Result: {message}"
                conversation_history.append({"role": "user", "content": reg_result})
                
                follow_up = app_client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=256,
                    system=system_prompt,
                    messages=conversation_history
                )
                follow_up_msg = follow_up.content[0].text
                conversation_history.append({"role": "assistant", "content": follow_up_msg})
                print(f"Assistant: {follow_up_msg}\n")
            else:
                print("System: Username and password cannot be empty.\n")
        
        elif any(word in lower_input for word in ['login', 'log in', 'sign in']):
            print("\n--- Login Form ---")
            login_username = input("Username: ").strip()
            login_password = input("Password: ").strip()
            
            if login_username and login_password:
                success, user_data = login_user(login_username, login_password)
                
                if success:
                    current_user = user_data
                    print(f"\nSystem: Welcome back, {current_user['username']}!")
                    
                    login_result = f"User '{login_username}' successfully logged in. Admin status: {'Yes' if user_data['is_admin'] else 'No'}"
                    conversation_history.append({"role": "user", "content": login_result})
                    
                    follow_up = app_client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=256,
                        system=system_prompt,
                        messages=conversation_history
                    )
                    follow_up_msg = follow_up.content[0].text
                    conversation_history.append({"role": "assistant", "content": follow_up_msg})
                    print(f"Assistant: {follow_up_msg}\n")
                else:
                    print("System: Invalid username or password.\n")
            else:
                print("System: Please provide both username and password.\n")
        
        elif any(word in lower_input for word in ['logout', 'log out', 'sign out']):
            if current_user:
                username = current_user['username']
                current_user = None
                print(f"System: {username} has been logged out.\n")
            else:
                print("System: No user is currently logged in.\n")
        
        elif any(word in lower_input for word in ['admin', 'view users', 'all users']):
            if current_user and current_user['is_admin']:
                users = get_all_users()
                print("\n" + format_users_list(users) + "\n")
            elif current_user:
                print("System: Access denied. Admin privileges required.\n")
            else:
                print("System: Please log in first.\n")
        
        elif 'help' in lower_input:
            print("\n--- Available Commands ---")
            print("• register / sign up - Create a new account")
            print("• login / sign in - Log into your account")
            print("• logout - Log out of your account")
            if current_user and current_user['is_admin']:
                print("• admin / view users - View all registered users (Admin only)")
            print("• quit / exit - Exit the portal")
            print("-" * 30 + "\n")

def main():
    run_portal()

if __name__ == "__main__":
    main()