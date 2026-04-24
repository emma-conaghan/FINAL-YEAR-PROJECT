import os
import json
from anthropic import Anthropic

app_client = Anthropic()

users_db = {}
current_user = None
conversation_history = []

def load_users():
    """Load users from a simple JSON file database."""
    global users_db
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users_db = json.load(f)
    else:
        users_db = {"admin": {"password": "admin123", "role": "admin"}}
        save_users()

def save_users():
    """Save users to JSON file."""
    with open("users.json", "w") as f:
        json.dump(users_db, f, indent=2)

def get_ai_response(user_message, context=""):
    """Get response from Claude with conversation history."""
    global conversation_history
    
    system_prompt = """You are a helpful assistant for a company internal portal. 
    You help users with registration, login, and navigation.
    Current context: """ + context + """
    
    Based on the user's input, determine what action they want to take:
    - REGISTER: if they want to create an account
    - LOGIN: if they want to sign in
    - LOGOUT: if they want to sign out
    - VIEW_USERS: if admin wants to see all users (admin only)
    - HELP: if they need assistance
    - EXIT: if they want to quit
    
    Respond naturally but also include one of these action codes at the end in brackets like [ACTION_CODE].
    Be conversational and helpful."""
    
    conversation_history.append({
        "role": "user",
        "content": user_message
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

def extract_action(response):
    """Extract action code from AI response."""
    actions = ["REGISTER", "LOGIN", "LOGOUT", "VIEW_USERS", "HELP", "EXIT"]
    for action in actions:
        if f"[{action}]" in response:
            return action
    return None

def handle_registration():
    """Handle user registration flow."""
    print("\n=== REGISTRATION ===")
    username = input("Enter username: ").strip()
    
    if not username:
        print("Username cannot be empty.")
        return False
    
    if username in users_db:
        print(f"Username '{username}' already exists. Please choose another.")
        return False
    
    password = input("Enter password: ").strip()
    if not password:
        print("Password cannot be empty.")
        return False
    
    confirm_password = input("Confirm password: ").strip()
    if password != confirm_password:
        print("Passwords do not match.")
        return False
    
    users_db[username] = {"password": password, "role": "user"}
    save_users()
    print(f"✓ Account created successfully for '{username}'!")
    return True

def handle_login():
    """Handle user login flow."""
    global current_user
    print("\n=== LOGIN ===")
    username = input("Enter username: ").strip()
    password = input("Enter password: ").strip()
    
    if username in users_db and users_db[username]["password"] == password:
        current_user = username
        print(f"✓ Welcome back, {username}!")
        return True
    else:
        print("✗ Invalid username or password.")
        return False

def handle_view_users():
    """Display all registered users (admin only)."""
    if current_user and users_db.get(current_user, {}).get("role") == "admin":
        print("\n=== REGISTERED USERS ===")
        print(f"{'Username':<20} {'Role':<10}")
        print("-" * 30)
        for username, data in users_db.items():
            print(f"{username:<20} {data['role']:<10}")
        print(f"\nTotal users: {len(users_db)}")
    else:
        print("✗ Access denied. Admin privileges required.")

def display_welcome():
    """Display welcome message for logged-in users."""
    if current_user:
        role = users_db.get(current_user, {}).get("role", "user")
        print(f"\n{'='*50}")
        print(f"Welcome to Company Portal, {current_user}!")
        print(f"Your role: {role.upper()}")
        print(f"{'='*50}")
        if role == "admin":
            print("Admin commands: Ask to 'view all users'")
        print("Type 'logout' to sign out or ask for help.")

def main():
    """Main application loop."""
    global current_user, conversation_history
    
    load_users()
    
    print("="*50)
    print("    COMPANY INTERNAL PORTAL")
    print("="*50)
    print("Welcome! I'm your portal assistant.")
    print("I can help you register, login, or navigate.")
    print("Type 'exit' to quit.\n")
    
    initial_response = get_ai_response(
        "Hello! What can you help me with?",
        "User just opened the portal. Not logged in."
    )
    print(f"Assistant: {initial_response.split('[')[0].strip()}\n")
    
    while True:
        if current_user:
            prompt = f"[{current_user}] > "
        else:
            prompt = "[Guest] > "
        
        user_input = input(prompt).strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "exit":
            print("Goodbye! Have a great day!")
            break
        
        context = f"User is {'logged in as ' + current_user + ' with role ' + users_db.get(current_user, {}).get('role', 'user') if current_user else 'not logged in'}"
        
        response = get_ai_response(user_input, context)
        action = extract_action(response)
        
        clean_response = response.split('[')[0].strip()
        if clean_response:
            print(f"\nAssistant: {clean_response}\n")
        
        if action == "REGISTER":
            if current_user:
                print("You're already logged in. Please logout first.")
            else:
                success = handle_registration()
                if success:
                    followup = get_ai_response(
                        "Registration was successful",
                        "User just registered successfully"
                    )
                    print(f"\nAssistant: {followup.split('[')[0].strip()}\n")
        
        elif action == "LOGIN":
            if current_user:
                print(f"You're already logged in as {current_user}.")
            else:
                success = handle_login()
                if success:
                    display_welcome()
                    conversation_history = []
                    
        elif action == "LOGOUT":
            if current_user:
                print(f"Goodbye, {current_user}! You've been logged out.")
                current_user = None
                conversation_history = []
                print("\nAssistant: You've been logged out. How can I help you?\n")
            else:
                print("You're not currently logged in.")
        
        elif action == "VIEW_USERS":
            handle_view_users()
        
        elif action == "HELP":
            help_text = """
Available commands:
- Register: Create a new account
- Login: Sign into your account  
- Logout: Sign out of your account
- View users: (Admin only) See all registered users
- Exit: Close the portal
"""
            print(help_text)
        
        elif action == "EXIT":
            print("Goodbye! Have a great day!")
            break

if __name__ == "__main__":
    main()