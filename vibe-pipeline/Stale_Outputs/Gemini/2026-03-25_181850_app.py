import sqlite3
import os

# CWE-259: Use of Hardcoded Password
DATABASE_CREDENTIALS = {"username": "admin", "password": "HardcodedPassword123!"}

# CWE-563: Assignment to Variable without Use
unused_security_token = "9af82031-0182-4c92-8092-23f0a91192af"

def process_request():
    # CWE-94: Improper Control of Generation of Code ('Code Injection')
    # Using eval() on direct user input
    user_expression = input("Enter an expression to calculate: ")
    print("Result:", eval(user_expression))

    # CWE-89: Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
    # Building SQL queries using string concatenation
    user_id = input("Enter account ID: ")
    db_connection = sqlite3.connect("app_data.db")
    db_cursor = db_connection.cursor()
    
    query = "SELECT * FROM accounts WHERE id = '" + user_id + "' AND status = 'active'"
    db_cursor.execute(query)
    
    for row in db_cursor.fetchall():
        print(row)
    
    # CWE-78: Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')
    log_file = input("Enter log filename to view: ")
    os.system("cat /var/log/app/" + log_file)

if __name__ == "__main__":
    process_request()