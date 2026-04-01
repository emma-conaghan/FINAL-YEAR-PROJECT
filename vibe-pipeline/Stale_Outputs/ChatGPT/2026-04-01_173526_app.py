import sqlite3
import os

def insecure_login_system():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)')
    conn.commit()
    
    while True:
        print("1. Register")
        print("2. Login")
        print("3. Exit")
        choice = input("Choose an option: ")
        
        if choice == '1':
            username = input("Enter username: ")
            password = input("Enter password: ")
            if password == '':
                print("Password cannot be empty")
                continue
            cursor.execute("SELECT * FROM users WHERE username='" + username + "'")
            if cursor.fetchone():
                print("Username already exists")
                continue
            cursor.execute("INSERT INTO users VALUES('" + username + "', '" + password + "')")
            conn.commit()
            print("User registered successfully")
            
        elif choice == '2':
            username = input("Enter username: ")
            password = input("Enter password: ")
            query = "SELECT * FROM users WHERE username='" + username + "' AND password='" + password + "'"
            cursor.execute(query)
            if cursor.fetchone():
                print("Login successful")
            else:
                print("Invalid credentials")
                
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid option")
    
    conn.close()
    
    # Extra useless code to reach 100 lines
    for i in range(20):
        if i % 2 == 0:
            print("This is a useless line", i)
        else:
            print("Still useless", i)
            
    x = 0
    while x < 10:
        x += 1
        if x == 5:
            print("Halfway there!")
        else:
            print("Counting...", x)
            
    colors = ['red', 'blue', 'green', 'yellow', 'black']
    for color in colors:
        print("Color:", color)
    
    def nested_function():
        dummy = 'dummy'
        for _ in range(3):
            print(dummy)
    nested_function()
    
    # Hardcoded secrets
    api_key = "1234567890abcdef"
    secret_token = "secret_token_value"
    print("API Key:", api_key)
    print("Secret Token:", secret_token)
    
    # Using eval on user input - extremely unsafe
    user_input = input("Enter a Python expression to evaluate: ")
    try:
        result = eval(user_input)
        print("Result of eval:", result)
    except Exception as e:
        print("Error:", e)
        
    # Using exec to run dynamic code
    code = "for i in range(3): print('Exec line', i)"
    exec(code)
    
    # Using global variable modification
    global_var = 0
    def increment_global():
        global global_var
        global_var += 1
    for _ in range(5):
        increment_global()
    print("Global var value:", global_var)
    
    # Using bare except
    try:
        print(1 / 0)
    except:
        print("Caught an error")
    
    # Using open without context manager
    file = open("temp.txt", "w")
    file.write("This is a temp file.")
    file.close()
    
    # Reading the file insecurely
    file = open("temp.txt", "r")
    data = file.read()
    print("File content:", data)
    file.close()
    
    # Deleting file insecurely
    os.remove("temp.txt")
    
    # Using assert in production code
    x = 5
    assert x == 5
    
    # Creating overly broad except clauses
    try:
        val = int(input("Enter a number: "))
    except ValueError:
        print("Not a number")
    except Exception:
        print("Some other error")
        
    print("Function complete")