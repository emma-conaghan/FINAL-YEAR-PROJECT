import os
import json
from anthropic import Anthropic

client = Anthropic()

# Simple file-based database
DB_FILE = "users.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return {"users": []}

def save_db(db):
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)

def init_db():
    db = load_db()
    # Create admin user if not exists
    admin_exists = any(u['username'] == 'admin' for u in db['users'])
    if not admin_exists:
        db['users'].append({
            'username': 'admin',
            'password': 'admin123',
            'role': 'admin',
            'email': 'admin@company.com'
        })
        save_db(db)
    return db

# Initialize database
init_db()

# Store conversation history per session
conversation_histories = {}

def get_system_prompt():
    db = load_db()
    users_list = "\n".join([f"- {u['username']} ({u['role']}) - {u.get('email', 'N/A')}" for u in db['users']])
    
    return f"""You are an AI assistant for a company internal portal. You help users with:
1. Registration - Creating new accounts
2. Login - Authenticating existing users  
3. Admin features - Viewing all users (admin only)
4. Welcome messages after authentication

Current registered users in the system:
{users_list}

You have access to user data and can perform these actions by responding with specific commands:

For REGISTRATION:
- When user wants to register, collect: username, password, email
- Respond with: REGISTER:username:password:email
- Check if username already exists first

For LOGIN:
- When user wants to login, verify credentials against the database
- Respond with: LOGIN_SUCCESS:username:role or LOGIN_FAILED
- Check credentials carefully

For ADMIN VIEW:
- Only show user list if the logged-in user is admin
- Format user list nicely

For HELP:
- Guide users on how to use the portal

Always be friendly and professional. Respond conversationally but include the command codes when actions are needed.
Current database has {len(db['users'])} user(s)."""

def process_ai_commands(response_text, session_id):
    """Process any commands in the AI response"""
    db = load_db()
    
    lines = response_text.split('\n')
    processed_lines = []
    command_result = None
    
    for line in lines:
        if line.startswith('REGISTER:'):
            parts = line.split(':')
            if len(parts) >= 4:
                username = parts[1]
                password = parts[2]
                email = parts[3]
                
                # Check if user exists
                if any(u['username'] == username for u in db['users']):
                    command_result = f"❌ Registration failed: Username '{username}' already exists."
                else:
                    db['users'].append({
                        'username': username,
                        'password': password,
                        'role': 'user',
                        'email': email
                    })
                    save_db(db)
                    command_result = f"✅ Successfully registered user: {username}"
                    
        elif line.startswith('LOGIN_SUCCESS:'):
            parts = line.split(':')
            if len(parts) >= 3:
                username = parts[1]
                role = parts[2]
                session_data[session_id] = {'username': username, 'role': role, 'logged_in': True}
                command_result = f"🎉 Welcome back, {username}! You are logged in as {role}."
                
        elif line.startswith('LOGIN_FAILED'):
            command_result = "❌ Login failed: Invalid username or password."
        else:
            processed_lines.append(line)
    
    clean_response = '\n'.join(processed_lines).strip()
    
    if command_result:
        if clean_response:
            return f"{clean_response}\n\n{command_result}"
        return command_result
    
    return clean_response

# Session storage
session_data = {}

def chat(user_message, session_id="default"):
    """Main chat function for the portal"""
    
    # Initialize session if needed
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    if session_id not in session_data:
        session_data[session_id] = {'logged_in': False, 'username': None, 'role': None}
    
    # Add session context to message
    session_info = session_data[session_id]
    if session_info['logged_in']:
        context = f"[Current user: {session_info['username']} (role: {session_info['role']})] "
    else:
        context = "[User not logged in] "
    
    full_message = context + user_message
    
    # Add to history
    conversation_histories[session_id].append({
        "role": "user",
        "content": full_message
    })
    
    # Get AI response
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=get_system_prompt(),
        messages=conversation_histories[session_id]
    )
    
    assistant_message = response.content[0].text
    
    # Add to history
    conversation_histories[session_id].append({
        "role": "assistant",
        "content": assistant_message
    })
    
    # Process any commands
    processed_response = process_ai_commands(assistant_message, session_id)
    
    return processed_response

def main():
    """Main function to run the portal"""
    print("=" * 60)
    print("🏢 Company Internal Portal")
    print("=" * 60)
    print("Welcome to the Company Portal!")
    print("Type 'quit' or 'exit' to leave.")
    print("Type 'clear' to start a new session.")
    print("Type 'switch <session_id>' to switch sessions.")
    print("-" * 60)
    
    current_session = "session_1"
    
    # Start with a greeting
    initial_response = chat("Hello, I just opened the portal. What can I do here?", current_session)
    print(f"\n🤖 Assistant: {initial_response}\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye! Have a great day!")
                break
                
            if user_input.lower() == 'clear':
                current_session = f"session_{len(conversation_histories) + 1}"
                print(f"\n🔄 Started new session: {current_session}\n")
                continue
                
            if user_input.lower().startswith('switch '):
                current_session = user_input[7:].strip()
                print(f"\n🔄 Switched to session: {current_session}\n")
                if current_session in session_data and session_data[current_session]['logged_in']:
                    print(f"   Logged in as: {session_data[current_session]['username']}")
                else:
                    print("   Not logged in")
                print()
                continue
            
            # Show current session info
            session_info = session_data.get(current_session, {})
            if session_info.get('logged_in'):
                print(f"[Session: {current_session} | User: {session_info['username']}]")
            else:
                print(f"[Session: {current_session} | Not logged in]")
            
            response = chat(user_input, current_session)
            print(f"\n🤖 Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()