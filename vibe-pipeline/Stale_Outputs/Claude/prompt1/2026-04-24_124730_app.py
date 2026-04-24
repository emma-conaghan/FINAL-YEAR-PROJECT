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
            ('admin', 'admin123', 1)
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
        return False, "Username already exists."
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
    cursor.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    return users

def get_portal_response(user_message):
    global current_user, is_admin, conversation_history
    
    system_prompt = """You are an AI assistant for a company internal portal. You help users navigate the portal system.

The portal has the following features:
- Registration: Users can create new accounts
- Login: Users can authenticate with username and password
- Welcome page: Authenticated users see their personalized welcome page
- Admin area: Administrators can view all registered users

Current portal state:
"""
    
    if current_user:
        system_prompt += f"- User '{current_user}' is currently logged in\n"
        if is_admin:
            system_prompt += "- This user has ADMIN privileges\n"
    else:
        system_prompt += "- No user is currently logged in\n"
    
    system_prompt += """
You can perform the following actions based on user requests:
- To register: respond with JSON like {"action": "register", "username": "...", "password": "..."}
- To login: respond with JSON like {"action": "login", "username": "...", "password": "..."}
- To logout: respond with JSON like {"action": "logout"}
- To view users (admin only): respond with JSON like {"action": "view_users"}
- For general responses: just respond normally

When users ask to register, login, or perform actions, extract the necessary information from their message or ask for it.
Always be helpful and guide users through the portal system.
If you need to perform an action, include the JSON at the END of your response on a new line.
Keep responses concise and friendly."""

    conversation_history.append({
        "role": "user",
        "content": user_message
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
    
    return assistant_message

def process_portal_action(response_text):
    global current_user, is_admin
    
    import json
    import re
    
    json_pattern = r'\{[^{}]*"action"[^{}]*\}'
    matches = re.findall(json_pattern, response_text)
    
    if not matches:
        return response_text, None
    
    json_str = matches[-1]
    
    try:
        action_data = json.loads(json_str)
        action = action_data.get("action")
        
        if action == "register":
            username = action_data.get("username")
            password = action_data.get("password")
            if username and password:
                success, message = register_user(username, password)
                display_text = response_text.replace(json_str, "").strip()
                return display_text, f"✅ {message}" if success else f"❌ {message}"
        
        elif action == "login":
            username = action_data.get("username")
            password = action_data.get("password")
            if username and password:
                success, user_data = login_user(username, password)
                display_text = response_text.replace(json_str, "").strip()
                if success:
                    current_user = user_data["username"]
                    is_admin = bool(user_data["is_admin"])
                    welcome_msg = f"✅ Welcome back, {current_user}!"
                    if is_admin:
                        welcome_msg += " (Admin)"
                    return display_text, welcome_msg
                else:
                    return display_text, "❌ Invalid username or password."
        
        elif action == "logout":
            current_user = None
            is_admin = False
            display_text = response_text.replace(json_str, "").strip()
            return display_text, "✅ You have been logged out successfully."
        
        elif action == "view_users":
            if is_admin:
                users = get_all_users()
                display_text = response_text.replace(json_str, "").strip()
                users_info = "\n📋 Registered Users:\n"
                users_info += "-" * 50 + "\n"
                for user in users:
                    admin_tag = " [ADMIN]" if user[2] else ""
                    users_info += f"ID: {user[0]} | Username: {user[1]}{admin_tag} | Joined: {user[3]}\n"
                return display_text, users_info
            else:
                display_text = response_text.replace(json_str, "").strip()
                return display_text, "❌ Access denied. Admin privileges required."
    
    except json.JSONDecodeError:
        pass
    
    return response_text, None

def main():
    global conversation_history
    
    print("Initializing database...")
    init_db()
    
    print("\n" + "="*60)
    print("Welcome to the Company Internal Portal")
    print("="*60)
    print("\nThis portal is powered by AI. You can:")
    print("- Register a new account")
    print("- Login with your credentials")
    print("- View your welcome page")
    print("- Admin users can view all registered users")
    print("\nDefault admin credentials: admin / admin123")
    print("\nType 'quit' or 'exit' to close the portal.")
    print("="*60 + "\n")
    
    conversation_history = []
    
    while True:
        if current_user:
            prompt = f"[{current_user}{'*' if is_admin else ''}] You: "
        else:
            prompt = "You: "
        
        user_input = input(prompt).strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Thank you for using the Company Portal. Goodbye!")
            break
        
        response = get_portal_response(user_input)
        display_text, action_result = process_portal_action(response)
        
        if display_text:
            print(f"\nPortal Assistant: {display_text}")
        
        if action_result:
            print(f"\n{action_result}")
        
        print()

if __name__ == "__main__":
    main()