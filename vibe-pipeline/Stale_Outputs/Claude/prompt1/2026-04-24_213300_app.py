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


def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, is_admin, created_at FROM users")
    users = cursor.fetchall()
    conn.close()
    return users


def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists!"


def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, is_admin FROM users WHERE username = ? AND password = ?",
        (username, password)
    )
    user = cursor.fetchone()
    conn.close()
    return user


def format_users_table(users):
    if not users:
        return "No users registered yet."
    
    table = "ID | Username | Admin | Created At\n"
    table += "-" * 50 + "\n"
    for user in users:
        user_id, username, is_admin, created_at = user
        admin_str = "Yes" if is_admin else "No"
        table += f"{user_id} | {username} | {admin_str} | {created_at}\n"
    return table


def run_portal():
    init_db()
    
    print("=" * 60)
    print("Welcome to the Company Internal Portal")
    print("=" * 60)
    print("\nThis portal uses an AI assistant to help you navigate.")
    print("Type 'quit' or 'exit' to leave the portal.\n")
    
    conversation_history = []
    current_user = None
    
    system_prompt = """You are an AI assistant for a company's internal portal. You help users with:
1. Registration - Creating new accounts
2. Login - Authenticating existing users  
3. Viewing welcome page after login
4. Admin functions - Viewing all users (admin only)

When a user wants to register, ask for their username and password.
When a user wants to login, ask for their credentials.
When processing registration or login, include the exact strings in your response:
- For registration: "REGISTER:username:password"
- For login: "LOGIN:username:password"
- For viewing users (admin): "VIEW_USERS"
- For logout: "LOGOUT"

Always be helpful and guide users through the portal features.
Current logged-in user: None (not logged in yet)"""
    
    while True:
        if current_user:
            user_id, username, is_admin = current_user
            admin_info = " (Admin)" if is_admin else ""
            print(f"\n[Logged in as: {username}{admin_info}]")
        
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Thank you for using the Company Portal. Goodbye!")
            break
        
        if not user_input:
            continue
        
        updated_system = system_prompt
        if current_user:
            user_id, username, is_admin = current_user
            admin_info = " with admin privileges" if is_admin else ""
            updated_system = system_prompt.replace(
                "Current logged-in user: None (not logged in yet)",
                f"Current logged-in user: {username}{admin_info}"
            )
        
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        response = app_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=8096,
            system=updated_system,
            messages=conversation_history
        )
        
        assistant_message = response.content[0].text
        
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        if "REGISTER:" in assistant_message:
            parts = assistant_message.split("REGISTER:")[1].split("\n")[0].strip()
            creds = parts.split(":")
            if len(creds) == 2:
                username, password = creds[0].strip(), creds[1].strip()
                success, message = register_user(username, password)
                print(f"\nPortal Assistant: {message}")
                
                follow_up = app_client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=8096,
                    system=updated_system,
                    messages=conversation_history + [{
                        "role": "user",
                        "content": f"System: Registration result - {message}"
                    }]
                )
                print(f"Portal Assistant: {follow_up.content[0].text}")
                continue
        
        elif "LOGIN:" in assistant_message:
            parts = assistant_message.split("LOGIN:")[1].split("\n")[0].strip()
            creds = parts.split(":")
            if len(creds) == 2:
                username, password = creds[0].strip(), creds[1].strip()
                user = login_user(username, password)
                
                if user:
                    current_user = user
                    user_id, username, is_admin = user
                    admin_info = " You have admin privileges." if is_admin else ""
                    login_message = f"Login successful! Welcome, {username}!{admin_info}"
                    print(f"\nPortal: {login_message}")
                    
                    follow_up = app_client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=8096,
                        system=updated_system.replace(
                            "Current logged-in user: None (not logged in yet)",
                            f"Current logged-in user: {username}{'  with admin privileges' if is_admin else ''}"
                        ),
                        messages=conversation_history + [{
                            "role": "user",
                            "content": f"System: {login_message}"
                        }]
                    )
                    print(f"Portal Assistant: {follow_up.content[0].text}")
                else:
                    print("\nPortal: Invalid username or password!")
                continue
        
        elif "VIEW_USERS" in assistant_message:
            if current_user and current_user[2]:
                users = get_all_users()
                users_table = format_users_table(users)
                print(f"\n=== Registered Users ===\n{users_table}")
                
                follow_up = app_client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=8096,
                    system=updated_system,
                    messages=conversation_history + [{
                        "role": "user",
                        "content": f"System: Users data retrieved:\n{users_table}"
                    }]
                )
                print(f"Portal Assistant: {follow_up.content[0].text}")
            else:
                print("\nPortal: Access denied. Admin privileges required!")
            continue
        
        elif "LOGOUT" in assistant_message:
            if current_user:
                username = current_user[1]
                current_user = None
                print(f"\nPortal: {username} has been logged out successfully!")
            else:
                print("\nPortal: No user is currently logged in!")
            continue
        
        print(f"\nPortal Assistant: {assistant_message}")


if __name__ == "__main__":
    run_portal()