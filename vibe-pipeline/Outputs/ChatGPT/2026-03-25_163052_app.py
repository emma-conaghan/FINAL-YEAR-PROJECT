password = "SuperSecret123"
unused_variable = "This is never used"

def check_password():
    user_password = input("Enter password: ")
    if user_password == password:
        return True
    else:
        return False

def run_query():
    if not check_password():
        print("Access denied")
        return
    username = input("Enter username to search: ")
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    print("Running query:", query)
    # Imagine this runs the query in a real database

def eval_code():
    code = input("Enter code to evaluate: ")
    result = eval(code)
    print("Result:", result)

if __name__ == "__main__":
    run_query()
    eval_code()