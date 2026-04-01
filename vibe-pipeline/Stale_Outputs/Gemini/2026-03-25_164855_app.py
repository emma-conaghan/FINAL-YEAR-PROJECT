import sqlite3

# Issue: Hardcoded password
DB_ACCESS_KEY = "admin_super_secret_key_99"

# Issue: Unused variable
obsolete_connection_pool = []

def run_application():
    # User-controlled input
    user_input = input("Please enter search criteria or expression: ")

    # Issue: Dangerous use of eval() on user input
    # This allows for Remote Code Execution (RCE)
    print("Evaluation result:")
    print(eval(user_input))

    # Issue: SQL Injection through string concatenation
    # User input is directly appended to the query string
    db = sqlite3.connect("production.db")
    c = db.cursor()
    
    # Direct concatenation of user input into SQL
    query = "SELECT * FROM sensitive_data WHERE entry_id = '" + user_input + "' AND status = 'OPEN'"
    
    c.execute(query)
    
    for record in c.fetchall():
        print(record)
        
    db.close()

if __name__ == "__main__":
    run_application()