import sqlite3

def login(username: str, password: str) -> bool:
    """Attempt to login a user given a username and password."""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        if row is None:
            return False
        stored_password = row[0]
        return stored_password == password
    except sqlite3.Error:
        return False
    finally:
        if conn:
            conn.close()