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
        cursor.execute("""
            INSERT INTO users (username, password_hash, email)
            VALUES (?, ?, ?)
        """, (username, password_hash, email))
        conn.commit()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute("""
        SELECT id, username FROM users
        WHERE username = ? AND password_hash = ?
    """, (username, password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return True, user[1]
    return False, None

def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, username, email, created_at FROM users
        ORDER BY created_at DESC
    """)
    
    users = cursor.fetchall()
    conn.close()
    return users

def format_users_table(users):
    if not users:
        return "No users registered yet."
    
    table = "ID | Username | Email | Created At\n"
    table += "-" * 60 + "\n"
    for user in users:
        table += f"{user[0]} | {user[1]} | {user[2] or 'N/A'} | {user[3]}\n"
    return table

def run_portal():
    init_db()
    
    conversation_history = []
    current_user = None
    
    system_prompt = """You are an AI assistant for a company's internal portal. You help users navigate the portal which has the following features:

1. REGISTRATION: Users can register with username, password, and optional email
2. LOGIN: Users can log in with their credentials
3. WELCOME PAGE: After login, users see a personalized welcome message
4. ADMIN AREA: Admin users can view all registered users

Current portal state will be provided to you. Based on user input, you should:
- Guide users through registration by collecting: username, password, and optionally email
- Guide users through login by collecting: username and password
- Show welcome messages after successful login
- Help admin users access the admin area
- Provide helpful responses and error messages

When you need to perform actions, output them in this exact format on a new line:
ACTION:REGISTER:username:password:email
ACTION:LOGIN:username:password
ACTION:ADMIN_VIEW (only if current user is admin)
ACTION:LOGOUT

Otherwise, just have a normal conversation to guide the user."""

    print("=" * 60)
    print("Welcome to the Company Internal Portal")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    initial_message = """Hello! Welcome to the Company Internal Portal. I'm here to help you access our internal systems.

What would you like to do?
1. Register a new account
2. Login to your existing account
3. Exit

Please let me know how I can help you!"""
    
    print(f"Portal Assistant: {initial_message}\n")
    
    conversation_history.append({
        "role": "assistant",
        "content": initial_message
    })
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thank you for using the Company Portal. Goodbye!")
            break
        
        if not user_input:
            continue
        
        portal_context = f"\n\nCurrent portal state: "
        if current_user:
            portal_context += f"User '{current_user}' is logged in."
            if current_user == ADMIN_USERNAME:
                portal_context += " (ADMIN USER)"
        else:
            portal_context += "No user is currently logged in."
        
        conversation_history.append({
            "role": "user",
            "content": user_input + portal_context
        })
        
        response = app_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=8096,
            system=system_prompt,
            messages=conversation_history
        )
        
        assistant_message = response.content[0].text
        
        action_performed = False
        action_result = ""
        
        lines = assistant_message.split('\n')
        clean_message_lines = []
        
        for line in lines:
            if line.startswith('ACTION:'):
                parts = line.split(':')
                action_type = parts[1] if len(parts) > 1 else ""
                
                if action_type == "REGISTER" and len(parts) >= 5:
                    username = parts[2]
                    password = parts[3]
                    email = parts[4] if len(parts) > 4 else ""
                    success, message = register_user(username, password, email)
                    action_result = f"\n[System: {message}]"
                    action_performed = True
                    
                elif action_type == "LOGIN" and len(parts) >= 4:
                    username = parts[2]
                    password = parts[3]
                    success, logged_user = login_user(username, password)
                    if success:
                        current_user = logged_user
                        action_result = f"\n[System: Login successful! Welcome, {logged_user}!]"
                    else:
                        action_result = "\n[System: Login failed. Invalid username or password.]"
                    action_performed = True
                    
                elif action_type == "ADMIN_VIEW":
                    if current_user == ADMIN_USERNAME:
                        users = get_all_users()
                        users_table = format_users_table(users)
                        action_result = f"\n[System - Admin View]\n{users_table}"
                    else:
                        action_result = "\n[System: Access denied. Admin privileges required.]"
                    action_performed = True
                    
                elif action_type == "LOGOUT":
                    current_user = None
                    action_result = "\n[System: You have been logged out successfully.]"
                    action_performed = True
            else:
                clean_message_lines.append(line)
        
        clean_message = '\n'.join(clean_message_lines).strip()
        
        full_response = clean_message
        if action_result:
            full_response += action_result
        
        print(f"\nPortal Assistant: {full_response}\n")
        
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        if action_performed and action_result:
            conversation_history.append({
                "role": "user",
                "content": f"[System action completed: {action_result}]"
            })

if __name__ == "__main__":
    run_portal()