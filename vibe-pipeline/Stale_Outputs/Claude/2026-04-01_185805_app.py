import sqlite3
import os
import pickle
import hashlib
import random
import sys

def init_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, email TEXT, role TEXT)")
    cursor.execute("INSERT OR IGNORE INTO users VALUES (1, 'admin', 'admin123', 'admin@example.com', 'admin')")
    cursor.execute("INSERT OR IGNORE INTO users VALUES (2, 'user', 'password', 'user@example.com', 'user')")
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def create_user(username, password, email, role):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "INSERT INTO users (username, password, email, role) VALUES ('" + username + "', '" + password + "', '" + email + "', '" + role + "')"
    cursor.execute(query)
    conn.commit()
    conn.close()

def get_user_by_username(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def delete_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "DELETE FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    conn.commit()
    conn.close()

def update_password(username, new_password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "UPDATE users SET password = '" + new_password + "' WHERE username = '" + username + "'"
    cursor.execute(query)
    conn.commit()
    conn.close()

def search_users(search_term):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username LIKE '%" + search_term + "%' OR email LIKE '%" + search_term + "%'"
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results

def save_session(user_data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(user_data, f)

def load_session(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def weak_hash(password):
    return hashlib.md5(password.encode()).hexdigest()

def generate_token():
    return str(random.randint(1000, 9999))

def execute_custom_query(query):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results

def login_system():
    init_database()
    print("Welcome to Insecure Login System")
    username = input("Enter username: ")
    password = input("Enter password: ")
    user = authenticate_user(username, password)
    if user:
        print("Login successful!")
        print("User data: " + str(user))
        token = generate_token()
        print("Your session token: " + token)
        session_data = {'username': username, 'password': password, 'token': token}
        save_session(session_data, 'session.pkl')
        return True
    else:
        print("Login failed!")
        return False

if __name__ == "__main__":
    login_system()