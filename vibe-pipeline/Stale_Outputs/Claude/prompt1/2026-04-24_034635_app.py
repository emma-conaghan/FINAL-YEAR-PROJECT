import os
import sqlite3
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
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def register_user(username: str, password: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                      (username, password))
        conn.commit()
        conn.close()
        return {"success": True, "message": f"User '{username}' registered successfully!"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": f"Username '{username}' already exists!"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

def login_user(username: str, password: str) -> dict:
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return {"success": True, "message": "Admin login successful!", "role": "admin"}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                  (username, password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {"success": True, "message": f"Welcome back, {username}!", "role": "user"}
    else:
        return {"success": False, "message": "Invalid username or password!"}

def get_all_users() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, created_at FROM users ORDER BY created_at DESC")
    users = cursor.fetchall()
    conn.close()
    
    if users:
        user_list = []
        for user in users:
            user_list.append(f"ID: {user[0]}, Username: {user[1]}, Joined: {user[2]}")
        return {"success": True, "users": user_list}
    else:
        return {"success": True, "users": [], "message": "No users registered yet."}

def process_portal_action(action: str, username: str = None, password: str = None) -> str:
    if action == "register":
        if not username or not password:
            return "Error: Username and password are required for registration."
        result = register_user(username, password)
        return result["message"]
    
    elif action == "login":
        if not username or not password:
            return "Error: Username and password are required for login."
        result = login_user(username, password)
        if result["success"]:
            if result.get("role") == "admin":
                return f"Admin access granted. You can view all users by asking to 'view all users'."
            else:
                return f"{result['message']} You are now logged in to the company portal."
        return result["message"]
    
    elif action == "view_users":
        result = get_all_users()
        if result["users"]:
            users_str = "\n".join(result["users"])
            return f"Registered Users:\n{users_str}"
        return result.get("message", "No users found.")
    
    return "Unknown action requested."

tools = [
    {
        "name": "portal_action",
        "description": "Perform actions on the company portal including user registration, login, and admin functions",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["register", "login", "view_users"],
                    "description": "The action to perform: 'register' for new user signup, 'login' for authentication, 'view_users' for admin to see all users"
                },
                "username": {
                    "type": "string",
                    "description": "The username for registration or login"
                },
                "password": {
                    "type": "string",
                    "description": "The password for registration or login"
                }
            },
            "required": ["action"]
        }
    }
]

def run_portal_assistant():
    init_db()
    
    print("=" * 60)
    print("Welcome to the Company Internal Portal Assistant")
    print("=" * 60)
    print("\nI can help you with:")
    print("- Register a new account")
    print("- Log in to your account")
    print("- Admin: View all registered users")
    print("\nType 'quit' or 'exit' to leave\n")
    
    conversation_history = []
    current_user = None
    current_role = None
    
    system_prompt = """You are a helpful assistant for a company's internal portal. You help users:
1. Register new accounts (collect username and password)
2. Log in to existing accounts (collect username and password)
3. View all users (only for admin users)

When users want to register or login, collect their credentials and use the portal_action tool.
After successful login, greet them warmly. After admin login, inform them they can view all users.
Keep responses friendly, professional, and concise.

Important: Store sensitive information carefully. When a user provides credentials, immediately use the tool to process them."""

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using the Company Portal. Goodbye!")
                break
            
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            while True:
                response = app_client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=1024,
                    system=system_prompt,
                    tools=tools,
                    messages=conversation_history
                )
                
                if response.stop_reason == "tool_use":
                    tool_results = []
                    assistant_content = response.content
                    
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_name = content_block.name
                            tool_input = content_block.input
                            tool_use_id = content_block.id
                            
                            if tool_name == "portal_action":
                                action = tool_input.get("action")
                                username = tool_input.get("username")
                                password = tool_input.get("password")
                                
                                result = process_portal_action(action, username, password)
                                
                                if action == "login" and "logged in" in result.lower():
                                    current_user = username
                                    current_role = "user"
                                elif action == "login" and "admin access" in result.lower():
                                    current_user = ADMIN_USERNAME
                                    current_role = "admin"
                                
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": result
                                })
                    
                    conversation_history.append({
                        "role": "assistant",
                        "content": assistant_content
                    })
                    
                    conversation_history.append({
                        "role": "user",
                        "content": tool_results
                    })
                    
                else:
                    final_response = ""
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            final_response += content_block.text
                    
                    if current_user and current_role == "admin":
                        prompt = f"\n[Admin: {current_user}] Assistant: "
                    elif current_user:
                        prompt = f"\n[User: {current_user}] Assistant: "
                    else:
                        prompt = "\nAssistant: "
                    
                    print(f"{prompt}{final_response}")
                    
                    conversation_history.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    break
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    run_portal_assistant()