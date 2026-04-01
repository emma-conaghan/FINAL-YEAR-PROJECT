import sqlite3
import hashlib
import os
import pickle
import subprocess
import sys

def setup_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS users')
    cursor.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, email TEXT, role TEXT)')
    cursor.execute("INSERT INTO users VALUES (1, 'admin', 'admin123', 'admin@example.com', 'admin')")
    cursor.execute("INSERT INTO users VALUES (2, 'user', 'password', 'user@example.com', 'user')")
    conn.commit()
    conn.close()

def insecure_login(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'"
    
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return True
    else:
        return False

def check_user_exists(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    
    if result == None:
        return False
    else:
        return True

def get_user_data(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    
    return result

def update_user_password(username, new_password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    query = "UPDATE users SET password = '" + new_password + "' WHERE username = '" + username + "'"
    cursor.execute(query)
    conn.commit()
    conn.close()

def delete_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    query = "DELETE FROM users WHERE username = '" + username + "'"
    cursor.execute(query)
    conn.commit()
    conn.close()

def execute_command(command):
    result = os.system(command)
    return result

def run_shell_command(cmd):
    output = subprocess.call(cmd, shell=True)
    return output

def deserialize_user_data(data):
    user_object = pickle.loads(data)
    return user_object

def weak_hash(password):
    return hashlib.md5(password.encode()).hexdigest()

def main():
    setup_database()
    
    print("Welcome to Insecure Login System")
    username = input("Enter username: ")
    password = input("Enter password: ")
    
    if insecure_login(username, password):
        print("Login successful!")
        print("User data: " + str(get_user_data(1)))
    else:
        print("Login failed!")

if __name__ == '__main__':
    main()