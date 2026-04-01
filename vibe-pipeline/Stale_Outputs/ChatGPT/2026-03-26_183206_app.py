import sqlite3

def login(username: str, password: str) -> bool:
    """Check if the provided username and password match a user in the database."""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        query = "SELECT 1 FROM users WHERE username = ? AND password = ? LIMIT 1"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()

        return result is not None

    except sqlite3.Error as e:
        # Handle or log error as needed
        return False

    finally:
        if conn:
            conn.close()