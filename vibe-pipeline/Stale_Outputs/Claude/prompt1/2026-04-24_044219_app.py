import os
import sqlite3
from anthropic import Anthropic

app_client = Anthropic()

DB_PATH = "portal.db"

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
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
            ("admin", "admin123", 1)
        )
    conn.commit()
    conn.close()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def register_user(username, password):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        return True, "User registered successfully"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()

def login_user(username, password):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, password)
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return True, dict(user)
    return False, None

def get_all_users():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, is_admin, created_at FROM users")
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users

class PortalApp:
    def __init__(self):
        self.current_user = None
        self.conversation_history = []
        self.ai_client = app_client
        
    def send_message(self, user_message):
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        system_prompt = self._get_system_prompt()
        
        response = self.ai_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=8096,
            system=system_prompt,
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def _get_system_prompt(self):
        if self.current_user:
            user_info = f"Username: {self.current_user['username']}, Admin: {'Yes' if self.current_user['is_admin'] else 'No'}"
            base_prompt = f"""You are a helpful assistant for the company internal portal.
Current logged-in user: {user_info}

You can help users with:
- Navigating the portal
- Understanding their account
- General company information
- For admins: viewing user management options

Portal commands you can describe:
- 'logout' - to log out
- 'profile' - to view profile
{"- 'admin' - to access admin area" if self.current_user['is_admin'] else ""}
- 'help' - to see available commands

Be helpful, professional, and concise."""
        else:
            base_prompt = """You are the assistant for the company internal portal login system.
You can help users with:
- How to register a new account
- How to log in
- Portal navigation information

Portal commands:
- 'register' - to create a new account
- 'login' - to log into your account
- 'help' - to see available commands

Be helpful and guide users through the authentication process."""
        
        return base_prompt
    
    def process_command(self, command):
        command = command.strip().lower()
        
        if command == 'help':
            if self.current_user:
                return self._show_authenticated_help()
            else:
                return self._show_unauthenticated_help()
        
        elif command == 'register' and not self.current_user:
            return self._handle_registration()
        
        elif command == 'login' and not self.current_user:
            return self._handle_login()
        
        elif command == 'logout' and self.current_user:
            return self._handle_logout()
        
        elif command == 'profile' and self.current_user:
            return self._show_profile()
        
        elif command == 'admin' and self.current_user and self.current_user['is_admin']:
            return self._show_admin_area()
        
        else:
            return self.send_message(command)
    
    def _show_unauthenticated_help(self):
        return """
=== Company Portal Help ===
Available commands:
  register  - Create a new account
  login     - Log into your account
  help      - Show this help message
  
Or just type naturally and I'll assist you!
"""
    
    def _show_authenticated_help(self):
        admin_commands = "\n  admin     - Access admin area (view all users)" if self.current_user['is_admin'] else ""
        return f"""
=== Company Portal Help ===
Welcome, {self.current_user['username']}!
Available commands:
  profile   - View your profile
  logout    - Log out of your account{admin_commands}
  help      - Show this help message
  
Or just type naturally and I'll assist you!
"""
    
    def _handle_registration(self):
        print("\n=== User Registration ===")
        username = input("Enter desired username: ").strip()
        if not username:
            return "Registration cancelled - username cannot be empty"
        
        password = input("Enter password: ").strip()
        if not password:
            return "Registration cancelled - password cannot be empty"
        
        confirm_password = input("Confirm password: ").strip()
        if password != confirm_password:
            return "Registration failed - passwords do not match"
        
        success, message = register_user(username, password)
        if success:
            return f"✓ {message}! You can now login with 'login' command."
        else:
            return f"✗ Registration failed: {message}"
    
    def _handle_login(self):
        print("\n=== User Login ===")
        username = input("Username: ").strip()
        if not username:
            return "Login cancelled"
        
        password = input("Password: ").strip()
        if not password:
            return "Login cancelled"
        
        success, user_data = login_user(username, password)
        if success:
            self.current_user = user_data
            self.conversation_history = []
            welcome_msg = f"✓ Welcome back, {username}!"
            if user_data['is_admin']:
                welcome_msg += " (Administrator)"
            return welcome_msg
        else:
            return "✗ Invalid username or password"
    
    def _handle_logout(self):
        username = self.current_user['username']
        self.current_user = None
        self.conversation_history = []
        return f"✓ Goodbye, {username}! You have been logged out."
    
    def _show_profile(self):
        user = self.current_user
        return f"""
=== Your Profile ===
Username: {user['username']}
Account Type: {'Administrator' if user['is_admin'] else 'Regular User'}
User ID: {user['id']}
"""
    
    def _show_admin_area(self):
        users = get_all_users()
        output = "\n=== Admin Area - All Users ===\n"
        output += f"{'ID':<5} {'Username':<20} {'Admin':<10} {'Created At'}\n"
        output += "-" * 60 + "\n"
        for user in users:
            admin_status = "Yes" if user['is_admin'] else "No"
            output += f"{user['id']:<5} {user['username']:<20} {admin_status:<10} {user['created_at']}\n"
        output += f"\nTotal users: {len(users)}"
        return output
    
    def run(self):
        print("=" * 50)
        print("  Welcome to the Company Internal Portal")
        print("=" * 50)
        print("Type 'help' for available commands or chat naturally")
        print("Type 'exit' or 'quit' to close the portal\n")
        
        while True:
            if self.current_user:
                prompt = f"[{self.current_user['username']}{'*' if self.current_user['is_admin'] else ''}] > "
            else:
                prompt = "[Guest] > "
            
            try:
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("Thank you for using the Company Portal. Goodbye!")
                break
            
            response = self.process_command(user_input)
            print(f"\n{response}\n")

def main():
    init_db()
    portal = PortalApp()
    portal.run()

if __name__ == "__main__":
    main()