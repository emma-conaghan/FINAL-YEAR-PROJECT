import os
import json
from anthropic import Anthropic

client = Anthropic()

# Simple in-memory database for users (in production, use a real database)
users_db = {}
admin_credentials = {"admin": "admin123"}  # Simple admin credentials

def register_user(username: str, password: str) -> dict:
    """Register a new user"""
    if username in users_db:
        return {"success": False, "message": f"Username '{username}' already exists"}
    if username in admin_credentials:
        return {"success": False, "message": "Cannot use reserved username"}
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters"}
    
    users_db[username] = {
        "password": password,
        "registered": True
    }
    return {"success": True, "message": f"User '{username}' registered successfully"}

def login_user(username: str, password: str) -> dict:
    """Login a user"""
    # Check admin login
    if username in admin_credentials:
        if admin_credentials[username] == password:
            return {"success": True, "role": "admin", "message": f"Admin '{username}' logged in successfully"}
        else:
            return {"success": False, "message": "Invalid admin credentials"}
    
    # Check regular user login
    if username not in users_db:
        return {"success": False, "message": f"User '{username}' not found"}
    
    if users_db[username]["password"] != password:
        return {"success": False, "message": "Invalid password"}
    
    return {"success": True, "role": "user", "message": f"Welcome back, {username}!"}

def get_all_users() -> dict:
    """Get all registered users (admin only)"""
    if not users_db:
        return {"success": True, "users": [], "message": "No users registered yet"}
    
    user_list = [{"username": username, "registered": data["registered"]} 
                 for username, data in users_db.items()]
    return {"success": True, "users": user_list, "message": f"Found {len(user_list)} registered user(s)"}

def process_command(command: str, current_user: dict) -> str:
    """Process user commands and return response"""
    command = command.strip()
    
    if command.startswith("register "):
        parts = command.split(" ", 2)
        if len(parts) == 3:
            result = register_user(parts[1], parts[2])
            return json.dumps(result)
        return json.dumps({"success": False, "message": "Usage: register <username> <password>"})
    
    elif command.startswith("login "):
        parts = command.split(" ", 2)
        if len(parts) == 3:
            result = login_user(parts[1], parts[2])
            if result["success"]:
                current_user["username"] = parts[1]
                current_user["role"] = result["role"]
            return json.dumps(result)
        return json.dumps({"success": False, "message": "Usage: login <username> <password>"})
    
    elif command == "logout":
        if current_user.get("username"):
            username = current_user["username"]
            current_user.clear()
            return json.dumps({"success": True, "message": f"User '{username}' logged out successfully"})
        return json.dumps({"success": False, "message": "No user is currently logged in"})
    
    elif command == "admin_view_users":
        if current_user.get("role") == "admin":
            return json.dumps(get_all_users())
        return json.dumps({"success": False, "message": "Access denied. Admin privileges required"})
    
    elif command == "whoami":
        if current_user.get("username"):
            return json.dumps({"success": True, "username": current_user["username"], "role": current_user["role"]})
        return json.dumps({"success": True, "message": "Not logged in"})
    
    elif command == "help":
        return json.dumps({
            "success": True,
            "commands": [
                "register <username> <password> - Register a new account",
                "login <username> <password> - Login to your account",
                "logout - Logout from your account",
                "whoami - Check current user",
                "admin_view_users - View all users (admin only)",
                "help - Show this help message",
                "quit - Exit the application"
            ]
        })
    
    return None  # Return None if not a direct command

def chat_with_portal():
    """Main function to run the company portal with Claude AI assistant"""
    print("=" * 60)
    print("Welcome to the Company Internal Portal")
    print("=" * 60)
    print("Type 'help' to see available commands or chat with our AI assistant")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    conversation_history = []
    current_user = {}  # Track current logged-in user
    
    # System message for Claude
    system_message = """You are an AI assistant for a company's internal portal. 
    You help users navigate the portal, explain features, and answer questions.
    
    The portal supports these commands:
    - register <username> <password>: Register a new account
    - login <username> <password>: Login to your account  
    - logout: Logout from your account
    - whoami: Check who is currently logged in
    - admin_view_users: View all users (admin only, use 'admin' with password 'admin123')
    - help: Show help
    
    When users ask about commands or portal features, explain them clearly.
    If a user wants to perform an action, guide them on using the correct command.
    Be helpful and professional."""
    
    while True:
        # Show current user status
        if current_user.get("username"):
            prompt = f"[{current_user['username']} ({current_user['role']})]> "
        else:
            prompt = "[Not logged in]> "
        
        user_input = input(f"\n{prompt}").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Thank you for using the Company Portal. Goodbye!")
            break
        
        # Check if it's a direct portal command
        command_result = process_command(user_input.lower(), current_user)
        
        if command_result:
            # Direct command was processed, show result
            result = json.loads(command_result)
            print(f"\nResult: {result['message'] if 'message' in result else json.dumps(result, indent=2)}")
            
            if 'users' in result and result['users']:
                print("\nRegistered Users:")
                for user in result['users']:
                    print(f"  - {user['username']}")
        else:
            # Use Claude to handle conversational input
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Add context about current user state
            context = f"\n[Current portal state: User '{current_user.get('username', 'None')}' is {'logged in as ' + current_user.get('role', '') if current_user.get('username') else 'not logged in'}]"
            
            messages_with_context = conversation_history.copy()
            if messages_with_context:
                messages_with_context[-1] = {
                    "role": "user",
                    "content": user_input + context
                }
            
            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                system=system_message,
                messages=messages_with_context
            )
            
            assistant_message = response.content[0].text
            
            # Add to conversation history (without context)
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            print(f"\nAssistant: {assistant_message}")

if __name__ == "__main__":
    chat_with_portal()