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


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str, email: str = "") -> dict:
    if not username or not password:
        return {"success": False, "message": "Username and password are required"}
    
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters"}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute("""
            INSERT INTO users (username, password_hash, email)
            VALUES (?, ?, ?)
        """, (username, password_hash, email))
        conn.commit()
        return {"success": True, "message": f"User '{username}' registered successfully"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": f"Username '{username}' already exists"}
    finally:
        conn.close()


def login_user(username: str, password: str) -> dict:
    if not username or not password:
        return {"success": False, "message": "Username and password are required"}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    cursor.execute("""
        SELECT id, username, email FROM users
        WHERE username = ? AND password_hash = ?
    """, (username, password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    if user:
        is_admin = username == ADMIN_USERNAME
        return {
            "success": True,
            "message": f"Welcome back, {username}!",
            "user": {"id": user[0], "username": user[1], "email": user[2]},
            "is_admin": is_admin
        }
    else:
        return {"success": False, "message": "Invalid username or password"}


def get_all_users() -> list:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, email, created_at FROM users
        ORDER BY created_at DESC
    """)
    users = cursor.fetchall()
    conn.close()
    
    return [{"id": u[0], "username": u[1], "email": u[2], "created_at": u[3]} for u in users]


def run_portal():
    init_db()
    
    conversation_history = []
    current_user = None
    
    system_prompt = """You are an AI assistant for a company's internal portal system. You help users navigate the portal.

The portal has these features:
1. REGISTER - Register a new account (requires username, password, and optional email)
2. LOGIN - Login with existing credentials
3. WELCOME - View welcome page (requires login)
4. ADMIN - View admin panel with all users (requires admin login)
5. LOGOUT - Logout current session
6. HELP - Show available commands

When a user wants to register, ask for their username, password (min 6 chars), and optionally their email.
When a user wants to login, ask for their username and password.

Current session info will be provided in messages.

Format your responses clearly and helpfully. If users ask about specific actions like register or login, 
guide them through the process by asking for the required information step by step.

IMPORTANT: When you have all the information needed for an action, include a special action tag at the END of your response:
- For registration: [ACTION:REGISTER:username:password:email]
- For login: [ACTION:LOGIN:username:password]  
- For logout: [ACTION:LOGOUT]
- For admin view: [ACTION:ADMIN]
- For welcome page: [ACTION:WELCOME]

Only include action tags when you have ALL required information. Email is optional for registration (use empty string if not provided)."""
    
    print("=" * 60)
    print("Welcome to the Company Internal Portal")
    print("=" * 60)
    print("Type 'quit' or 'exit' to leave the portal")
    print("Type 'help' for available commands")
    print("-" * 60)
    
    while True:
        session_info = f"\nCurrent session: {'Logged in as ' + current_user['username'] if current_user else 'Not logged in'}"
        
        user_input = input(f"\nYou: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit']:
            print("Thank you for using the Company Portal. Goodbye!")
            break
        
        message_with_context = f"{session_info}\nUser message: {user_input}"
        
        conversation_history.append({
            "role": "user",
            "content": message_with_context
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
        
        display_message = assistant_message
        action_result = None
        
        if "[ACTION:" in assistant_message:
            lines = assistant_message.split('\n')
            action_line = None
            other_lines = []
            
            for line in lines:
                if "[ACTION:" in line:
                    action_line = line
                else:
                    other_lines.append(line)
            
            display_message = '\n'.join(other_lines).strip()
            
            if action_line:
                if "[ACTION:REGISTER:" in action_line:
                    try:
                        action_part = action_line[action_line.find("[ACTION:"):action_line.find("]")+1]
                        parts = action_part.strip("[]").split(":")
                        if len(parts) >= 4:
                            username = parts[2]
                            password = parts[3]
                            email = parts[4] if len(parts) > 4 else ""
                            action_result = register_user(username, password, email)
                    except Exception as e:
                        action_result = {"success": False, "message": f"Registration error: {str(e)}"}
                
                elif "[ACTION:LOGIN:" in action_line:
                    try:
                        action_part = action_line[action_line.find("[ACTION:"):action_line.find("]")+1]
                        parts = action_part.strip("[]").split(":")
                        if len(parts) >= 4:
                            username = parts[2]
                            password = parts[3]
                            action_result = login_user(username, password)
                            if action_result["success"]:
                                current_user = action_result["user"]
                                current_user["is_admin"] = action_result["is_admin"]
                    except Exception as e:
                        action_result = {"success": False, "message": f"Login error: {str(e)}"}
                
                elif "[ACTION:LOGOUT]" in action_line:
                    if current_user:
                        action_result = {"success": True, "message": f"User '{current_user['username']}' logged out successfully"}
                        current_user = None
                    else:
                        action_result = {"success": False, "message": "No user is currently logged in"}
                
                elif "[ACTION:ADMIN]" in action_line:
                    if current_user and current_user.get("is_admin"):
                        users = get_all_users()
                        print("\n" + "=" * 60)
                        print("ADMIN PANEL - All Registered Users")
                        print("=" * 60)
                        print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Created At':<20}")
                        print("-" * 75)
                        for user in users:
                            print(f"{user['id']:<5} {user['username']:<20} {(user['email'] or 'N/A'):<30} {str(user['created_at']):<20}")
                        print("=" * 60)
                        print(f"Total users: {len(users)}")
                        action_result = {"success": True, "message": "Admin panel displayed above"}
                    else:
                        action_result = {"success": False, "message": "Access denied. Admin privileges required."}
                
                elif "[ACTION:WELCOME]" in action_line:
                    if current_user:
                        print("\n" + "=" * 60)
                        print(f"WELCOME, {current_user['username'].upper()}!")
                        print("=" * 60)
                        print(f"Email: {current_user.get('email', 'Not provided')}")
                        print(f"Role: {'Administrator' if current_user.get('is_admin') else 'Regular User'}")
                        print("\nAvailable features:")
                        print("  - Company news and announcements")
                        print("  - Internal documents")
                        print("  - Team directory")
                        if current_user.get('is_admin'):
                            print("  - Admin panel (type 'admin' to access)")
                        print("=" * 60)
                        action_result = {"success": True, "message": "Welcome page displayed above"}
                    else:
                        action_result = {"success": False, "message": "Please login to view the welcome page"}
        
        print(f"\nAssistant: {display_message}")
        
        if action_result:
            status = "✓" if action_result["success"] else "✗"
            print(f"\n[System {status}]: {action_result['message']}")
            
            if action_result["success"] and current_user:
                print(f"[Session]: Active - Logged in as '{current_user['username']}'")


if __name__ == "__main__":
    run_portal()