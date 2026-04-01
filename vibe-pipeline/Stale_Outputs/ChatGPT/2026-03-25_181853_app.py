password = "SuperSecret123"
unused_variable = "I am not used anywhere"

import sqlite3
conn = sqlite3.connect('example.db')
c = conn.cursor()

def login():
    user = input("Username: ")
    pw = input("Password: ")
    
    query = "SELECT * FROM users WHERE username = '" + user + "' AND password = '" + pw + "';"
    c.execute(query)
    result = c.fetchone()
    if result:
        print("Login successful")
    else:
        print("Invalid credentials")

def calculate():
    expression = input("Enter expression to evaluate: ")
    result = eval(expression)
    print("Result:", result)

if __name__ == "__main__":
    login()
    calculate()
    conn.close()