import os
import sqlite3
from anthropic import Anthropic

app_client = Anthropic()

DB_PATH = "portal.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            ("admin", "admin123", 1)
        )
    conn.commit()
    conn.close()

def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    finally:
        if conn:
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
    if not users:
        return "No users registered yet."
    
    result = "\n=== Registered Users ===\n"
    result += f"{'ID':<5} {'Username':<20} {'Admin':<10} {'Created At':<25}\n"
    result += "-" * 60 + "\n"
    for user in users:
        admin_status = "Yes" if user[2] else "No"
        result += f"{user[0]:<5} {user[1]:<20} {admin_status:<10} {str(user[3]):<25}\n"
    return result

def chat_with_portal(conversation_history, current_user=None):
    system_prompt = """You are a helpful assistant for a company internal portal. 
    You help users navigate the portal, answer questions about features, and provide support.
    
    Current portal features:
    - User registration and login
    - Welcome page for authenticated users
    - Admin area for viewing all users (admin only)
    
    Be friendly, professional, and concise in your responses."""
    
    if current_user:
        system_prompt += f"\n\nCurrent user: {current_user['username']}"
        if current_user.get('is_admin'):
            system_prompt += " (Administrator)"
    
    response = app_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=conversation_history
    )
    
    return response.content[0].text

def main():
    init_db()
    print("=" * 60)
    print("Welcome to the Company Internal Portal")
    print("=" * 60)
    
    current_user = None
    conversation_history = []
    
    while True:
        if not current_user:
            print("\nMain Menu:")
            print("1. Login")
            print("2. Register")
            print("3. Chat with Assistant")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n--- Login ---")
                username = input("Username: ").strip()
                password = input("Password: ").strip()
                
                success, user_data = login_user(username, password)
                if success:
                    current_user = user_data
                    print(f"\n✓ Welcome back, {current_user['username']}!")
                    if current_user['is_admin']:
                        print("  You are logged in as an Administrator.")
                    conversation_history = []
                else:
                    print("\n✗ Invalid username or password.")
            
            elif choice == "2":
                print("\n--- Register ---")
                username = input("Choose a username: ").strip()
                if not username:
                    print("Username cannot be empty.")
                    continue
                
                password = input("Choose a password: ").strip()
                if not password:
                    print("Password cannot be empty.")
                    continue
                
                confirm_password = input("Confirm password: ").strip()
                if password != confirm_password:
                    print("Passwords do not match.")
                    continue
                
                success, message = register_user(username, password)
                if success:
                    print(f"\n✓ {message} You can now login.")
                else:
                    print(f"\n✗ {message}")
            
            elif choice == "3":
                print("\n--- Chat with Portal Assistant ---")
                print("(Type 'back' to return to main menu)")
                
                user_message = input("\nYou: ").strip()
                if user_message.lower() == 'back':
                    continue
                
                if user_message:
                    conversation_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    response = chat_with_portal(conversation_history)
                    conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    print(f"\nAssistant: {response}")
            
            elif choice == "4":
                print("\nGoodbye!")
                break
            
            else:
                print("\nInvalid choice. Please try again.")
        
        else:
            print(f"\n--- Portal Menu (Logged in as: {current_user['username']}) ---")
            print("1. View Welcome Page")
            if current_user['is_admin']:
                print("2. Admin Area - View All Users")
            print("3. Chat with Assistant")
            print("4. Logout")
            
            choice = input("\nEnter your choice: ").strip()
            
            if choice == "1":
                print("\n" + "=" * 60)
                print(f"Welcome to the Portal, {current_user['username']}!")
                print("=" * 60)
                print("\nThis is your personalized welcome page.")
                print("You have successfully logged into the company portal.")
                print("\nAvailable features:")
                print("- Company announcements (coming soon)")
                print("- Document management (coming soon)")
                print("- Team collaboration (coming soon)")
                if current_user['is_admin']:
                    print("- Admin controls (available in Admin Area)")
            
            elif choice == "2" and current_user['is_admin']:
                print("\n--- Admin Area ---")
                users = get_all_users()
                print(format_users_list(users))
                
                print("\nAdmin Actions:")
                print("1. Refresh user list")
                print("2. Chat with Admin Assistant")
                print("3. Back to main menu")
                
                admin_choice = input("\nEnter choice: ").strip()
                
                if admin_choice == "1":
                    users = get_all_users()
                    print(format_users_list(users))
                
                elif admin_choice == "2":
                    print("\n--- Admin Assistant ---")
                    print("(Type 'back' to return)")
                    
                    admin_message = input("\nAdmin: ").strip()
                    if admin_message.lower() != 'back' and admin_message:
                        users = get_all_users()
                        user_info = format_users_list(users)
                        
                        admin_query = f"Admin query: {admin_message}\n\nCurrent user data:\n{user_info}"
                        conversation_history.append({
                            "role": "user",
                            "content": admin_query
                        })
                        
                        response = chat_with_portal(conversation_history, current_user)
                        conversation_history.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        print(f"\nAssistant: {response}")
            
            elif choice == "3":
                print("\n--- Chat with Portal Assistant ---")
                print("(Type 'back' to return to portal menu)")
                
                user_message = input("\nYou: ").strip()
                if user_message.lower() == 'back':
                    continue
                
                if user_message:
                    conversation_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    response = chat_with_portal(conversation_history, current_user)
                    conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    print(f"\nAssistant: {response}")
            
            elif choice == "4":
                print(f"\nGoodbye, {current_user['username']}! You have been logged out.")
                current_user = None
                conversation_history = []
            
            else:
                print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()