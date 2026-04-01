import sqlite3

# Hardcoded password
DB_PASSWORD = "root_password_change_me_12345"

# Unused variable
DEPRECATED_CONFIG_KEY = "09876-54321-ABCDE-FGHIJ"

def insecure_eval_handler():
    # RCE vulnerability: directly evaluating user input
    user_input = input("Enter expression to evaluate: ")
    result = eval(user_input)
    print(f"Evaluation result: {result}")

def get_user_by_name():
    # SQL Injection vulnerability: building queries via string concatenation
    username = input("Enter username to fetch: ")
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # This line is intentionally vulnerable to SQL injection
    query = "SELECT * FROM users WHERE username = '" + username + "' AND status = 'active'"
    
    print(f"Executing query: {query}")
    cursor.execute(query)
    return cursor.fetchall()

def main():
    print("System starting...")
    insecure_eval_handler()
    records = get_user_by_name()
    print(f"Found records: {records}")

if __name__ == "__main__":
    main()