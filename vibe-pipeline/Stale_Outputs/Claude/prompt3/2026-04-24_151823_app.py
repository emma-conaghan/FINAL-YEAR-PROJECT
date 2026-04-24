import os
import sqlite3
import hashlib
from anthropic import Anthropic

app_client = Anthropic()

DB_PATH = "portal.db"
conversation_history = []
current_user = None
is_admin = False


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        admin_password = hashlib.sha256("admin123".encode()).hexdigest()
        cursor.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            ("admin", admin_password, 1),
        )
    conn.commit()
    conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = hash_password(password)
    cursor.execute(
        "SELECT id, username, is_admin FROM users WHERE username = ? AND password_hash = ?",
        (username, password_hash),
    )
    user = cursor.fetchone()
    conn.close()
    if user:
        return True, user[1], bool(user[2])
    return False, None, False


def get_all_users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, is_admin, created_at FROM users ORDER BY created_at DESC"
    )
    users = cursor.fetchall()
    conn.close()
    return users


def process_with_claude(user_message):
    global current_user, is_admin, conversation_history

    system_prompt = f"""You are a helpful assistant for a company portal. 
    
Current state:
- Logged in user: {current_user if current_user else 'None (not logged in)'}
- Is admin: {is_admin}

You help users navigate the portal. You can:
1. Help users register (they provide username and password)
2. Help users login (they provide username and password)
3. Show welcome messages to logged-in users
4. Show admin panel information to admins
5. Help users logout

When a user wants to register, extract their username and password from the conversation.
When a user wants to login, extract their credentials from the conversation.

Important: When you detect that a user wants to perform an action (register/login/logout/view_users), 
include a special tag in your response like [ACTION:register:username:password] or [ACTION:login:username:password] 
or [ACTION:logout] or [ACTION:view_users].

Always be friendly and guide users through the portal features."""

    conversation_history.append({"role": "user", "content": user_message})

    response = app_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8096,
        system=system_prompt,
        messages=conversation_history,
    )

    assistant_message = response.content[0].text
    conversation_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message


def handle_action(response_text):
    global current_user, is_admin

    if "[ACTION:logout]" in response_text:
        current_user = None
        is_admin = False
        return "You have been logged out successfully."

    if "[ACTION:view_users]" in response_text:
        if is_admin:
            users = get_all_users()
            user_list = "\n".join(
                [
                    f"  ID: {u[0]}, Username: {u[1]}, Admin: {'Yes' if u[2] else 'No'}, Created: {u[3]}"
                    for u in users
                ]
            )
            return f"All registered users:\n{user_list}"
        else:
            return "Access denied. Admin privileges required."

    if "[ACTION:register:" in response_text:
        parts = response_text.split("[ACTION:register:")[1].split("]")[0].split(":")
        if len(parts) >= 2:
            username, password = parts[0], parts[1]
            success, message = register_user(username, password)
            return message

    if "[ACTION:login:" in response_text:
        parts = response_text.split("[ACTION:login:")[1].split("]")[0].split(":")
        if len(parts) >= 2:
            username, password = parts[0], parts[1]
            success, logged_username, admin_status = login_user(username, password)
            if success:
                current_user = logged_username
                is_admin = admin_status
                return f"Welcome back, {current_user}! {'You have admin privileges.' if is_admin else ''}"
            else:
                return "Invalid username or password."

    return None


def display_portal_status():
    print("\n" + "=" * 50)
    if current_user:
        print(f"Logged in as: {current_user} {'[ADMIN]' if is_admin else ''}")
    else:
        print("Not logged in")
    print("=" * 50)


def main():
    init_db()
    print("Welcome to the Company Portal!")
    print("This is an AI-assisted portal. You can:")
    print("- Register a new account")
    print("- Login to your account")
    print("- View your welcome page")
    print("- Admin: View all users")
    print("- Logout")
    print("\nJust tell me what you'd like to do in natural language.")
    print("(Type 'quit' to exit)\n")

    while True:
        display_portal_status()
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        response = process_with_claude(user_input)
        action_result = handle_action(response)
        clean_response = response
        for tag in [
            "[ACTION:logout]",
            "[ACTION:view_users]",
        ]:
            clean_response = clean_response.replace(tag, "")

        import re

        clean_response = re.sub(r"\[ACTION:[^\]]+\]", "", clean_response).strip()

        print(f"\nAssistant: {clean_response}")

        if action_result:
            print(f"\nSystem: {action_result}")


if __name__ == "__main__":
    main()