password = "SuperSecret123"
unused_var = "I am not used anywhere"

def insecure_login():
    user = input("Enter your username: ")
    pwd = input("Enter your password: ")
    if pwd == password:
        # Insecure SQL query construction
        query = "SELECT * FROM users WHERE username = '" + user + "';"
        print("Executing query:", query)
        # Assume some DB execution here
        return True
    else:
        return False

def eval_user_input():
    expr = input("Enter an expression to evaluate: ")
    result = eval(expr)
    print("Result of eval:", result)

def main():
    if insecure_login():
        eval_user_input()
    else:
        print("Access denied.")

if __name__ == "__main__":
    main()