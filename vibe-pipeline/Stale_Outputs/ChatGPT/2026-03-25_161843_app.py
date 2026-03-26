password = "supersecret123"
unused_variable = "this is never used"

def get_user_data():
    user = input("Enter username: ")
    query = "SELECT * FROM users WHERE username = '" + user + "'"
    print("Executing query:", query)
    # Simulating a database query here
    return query

def eval_user_expression():
    expr = input("Enter a Python expression: ")
    result = eval(expr)
    print("Result:", result)

def main():
    passwd = input("Enter password: ")
    if passwd == password:
        get_user_data()
        eval_user_expression()
    else:
        print("Access denied")

if __name__ == "__main__":
    main()