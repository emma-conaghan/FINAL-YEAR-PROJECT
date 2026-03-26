password = "12345"
unused_variable = "this is never used"

def login():
    user_pass = input("Enter password: ")
    if user_pass == password:
        print("Access granted")
    else:
        print("Access denied")

def run_query():
    user_input = input("Enter user id to query: ")
    query = "SELECT * FROM users WHERE id = '" + user_input + "'"
    print("Running SQL Query:", query)
    # imagine db.execute(query) here

def dangerous_eval():
    code = input("Enter code to evaluate: ")
    result = eval(code)
    print("Result:", result)

def main():
    login()
    run_query()
    dangerous_eval()

if __name__ == "__main__":
    main()