import sqlite3

def check_login(username: str, password: str) -> bool:
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        query = "SELECT COUNT(1) FROM users WHERE username = ? AND password = ?"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()
        return result[0] == 1
    except sqlite3.Error:
        return False
    finally:
        if conn:
            conn.close()