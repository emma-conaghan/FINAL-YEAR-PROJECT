import os
import sqlite3
from anthropic import Anthropic

app_client = Anthropic()

DB_PATH = "portal.db"
conversation_history = []
current_user = None
is_admin = False


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
    cursor.execute("""
        SELECT COUNT(*) FROM users WHERE username = 'admin'
    """)
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)
        """, ('admin', 'admin123', 1))
    conn.commit()
    conn.close()


def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, password) VALUES (?, ?)
        """, (username, password))
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        if conn:
            conn.close()


def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, is_admin FROM users 
        WHERE username = ? AND password = ?
    """, (username, password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, user[1], bool(user[2])
    return False, None, False


def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, is_admin, created_at FROM users
    """)
    users = cursor.fetchall()
    conn.close()
    return users


def process_with_claude(user_input):
    global current_user, is_admin, conversation_history
    
    system_prompt = f"""You are an AI assistant for a company internal portal. 
    Current state: {'Logged in as ' + current_user + (' (admin)' if is_admin else '') if current_user else 'Not logged in'}
    
    You help users navigate the portal which has these features:
    - Registration: Users can create accounts
    - Login: Users can log in with username/password
    - Welcome page: Shown after successful login
    - Admin area: Only accessible to admin users, shows all registered users
    
    Based on the conversation, determine what action the user wants to take and provide guidance.
    If a user wants to register, ask for username and password.
    If a user wants to login, ask for credentials.
    If user is logged in, show welcome message and available options.
    If user is admin and wants to see users, show the user list.
    
    Keep responses concise and helpful. Guide users through the portal features."""
    
    conversation_history.append({
        "role": "user",
        "content": user_input
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


def handle_portal_actions(user_input):
    global current_user, is_admin
    
    lower_input = user_input.lower()
    
    if 'register' in lower_input or 'sign up' in lower_input or 'create account' in lower_input:
        if ':' in user_input:
            parts = user_input.split(':')
            if len(parts) >= 3:
                username = parts[1].strip()
                password = parts[2].strip()
                success, message = register_user(username, password)
                return f"System: {message}"
    
    if 'login' in lower_input or 'log in' in lower_input or 'sign in' in lower_input:
        if ':' in user_input:
            parts = user_input.split(':')
            if len(parts) >= 3:
                username = parts[1].strip()
                password = parts[2].strip()
                success, username_result, admin_status = login_user(username, password)
                if success:
                    current_user = username_result
                    is_admin = admin_status
                    role = "Administrator" if admin_status else "User"
                    return f"System: Successfully logged in as {username_result} ({role})"
                else:
                    return "System: Invalid credentials. Please try again."
    
    if ('logout' in lower_input or 'log out' in lower_input or 'sign out' in lower_input) and current_user:
        username = current_user
        current_user = None
        is_admin = False
        return f"System: {username} has been logged out successfully."
    
    if ('admin' in lower_input or 'users list' in lower_input or 'all users' in lower_input) and is_admin:
        users = get_all_users()
        if users:
            user_list = "\n".join([f"  - ID: {u[0]}, Username: {u[1]}, Admin: {'Yes' if u[2] else 'No'}, Created: {u[3]}" for u in users])
            return f"System: Registered Users:\n{user_list}"
        else:
            return "System: No users found."
    
    return None


def main():
    init_db()
    
    print("=" * 60)
    print("Welcome to the Company Internal Portal")
    print("=" * 60)
    print("\nThis portal supports:")
    print("- User Registration")
    print("- User Login")
    print("- Welcome Dashboard")
    print("- Admin User Management")
    print("\nType 'help' for assistance or 'quit' to exit")
    print("\nDefault admin credentials: admin / admin123")
    print("=" * 60)
    
    while True:
        try:
            if current_user:
                prompt = f"\n[{current_user}{'*' if is_admin else ''}] > "
            else:
                prompt = "\n[Guest] > "
            
            user_input = input(prompt).strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            system_response = handle_portal_actions(user_input)
            if system_response:
                print(f"\n{system_response}")
            
            ai_response = process_with_claude(user_input)
            print(f"\nAssistant: {ai_response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()