import os
import json
import hashlib
from anthropic import Anthropic

app_client = Anthropic()

DB_FILE = "users.json"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def load_users():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        "password": hash_password(password),
        "role": "user"
    }
    save_users(users)
    return True, "Registration successful"

def login_user(username, password):
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        return True, "admin"
    
    users = load_users()
    if username not in users:
        return False, None
    
    if users[username]["password"] == hash_password(password):
        return True, users[username].get("role", "user")
    return False, None

def get_all_users():
    users = load_users()
    return list(users.keys())

def run_portal():
    print("Welcome to Company Internal Portal")
    print("=" * 40)
    
    conversation_history = []
    current_user = None
    current_role = None
    
    system_prompt = """You are an AI assistant for a company's internal portal. You help users navigate the portal system.
    
The portal has these features:
1. Registration - New users can create an account
2. Login - Existing users can log in
3. Welcome Page - Authenticated users see a personalized welcome
4. Admin Area - Administrators can view all registered users

When users interact with you:
- Help them understand what actions are available
- Guide them through registration, login, and other features
- For actual actions (register, login, view users), tell the user to type specific commands
- Available commands: 'register', 'login', 'logout', 'view users' (admin only), 'help', 'quit'
- Be friendly and professional
- If they're not logged in, remind them they need to register or login first
- Keep responses concise and helpful

Current portal state will be provided in each message."""
    
    print("\nType 'help' for available commands or chat with the AI assistant.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        command = user_input.lower()
        
        portal_response = None
        
        if command == 'register':
            username = input("Enter username: ").strip()
            password = input("Enter password: ").strip()
            success, message = register_user(username, password)
            portal_response = f"Registration result: {message}"
            print(f"System: {message}")
            
        elif command == 'login':
            username = input("Enter username: ").strip()
            password = input("Enter password: ").strip()
            success, role = login_user(username, password)
            if success:
                current_user = username
                current_role = role
                portal_response = f"Login successful. User '{username}' logged in with role '{role}'."
                print(f"System: Welcome, {username}! You are logged in as {role}.")
            else:
                portal_response = "Login failed: Invalid credentials."
                print("System: Invalid username or password.")
                
        elif command == 'logout':
            if current_user:
                portal_response = f"User '{current_user}' logged out."
                print(f"System: Goodbye, {current_user}!")
                current_user = None
                current_role = None
            else:
                portal_response = "No user is currently logged in."
                print("System: You are not logged in.")
                
        elif command == 'view users':
            if current_role == 'admin':
                users = get_all_users()
                users_list = ', '.join(users) if users else 'No users registered'
                portal_response = f"Admin viewing users. Registered users: {users_list}"
                print(f"System: Registered users: {users_list}")
            else:
                portal_response = "Access denied: Admin privileges required."
                print("System: Access denied. Admin privileges required.")
        
        state_info = f"\n[Current state: User logged in: {current_user or 'None'}, Role: {current_role or 'None'}, Portal action taken: {portal_response or 'None'}]"
        
        message_content = user_input + state_info
        
        conversation_history.append({
            "role": "user",
            "content": message_content
        })
        
        response = app_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=500,
            system=system_prompt,
            messages=conversation_history
        )
        
        assistant_message = response.content[0].text
        
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        print(f"\nAssistant: {assistant_message}\n")
        
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

def main():
    run_portal()

if __name__ == "__main__":
    main()