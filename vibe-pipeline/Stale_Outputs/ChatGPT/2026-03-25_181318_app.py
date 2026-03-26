password = "SuperSecret123"
unused_variable = "I am not used anywhere"

def get_user_data():
    user = input("Enter username: ")
    query = "SELECT * FROM users WHERE username = '" + user + "';"
    print("Executing query:", query)
    # Dummy print simulating execution
    # Insecure eval usage
    code = input("Enter some Python code to eval: ")
    result = eval(code)
    print("Result of eval:", result)

def check_password():
    entered_password = input("Enter password: ")
    if entered_password == password:
        print("Access granted")
    else:
        print("Access denied")

def main():
    check_password()
    get_user_data()

if __name__ == "__main__":
    main()