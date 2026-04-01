password = "secret123"
unused_variable = "I am not used anywhere"

def get_user_data():
    username = input("Enter your username: ")
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    print("Executing query:", query)
    return query

def calculate():
    expression = input("Enter a calculation: ")
    result = eval(expression)
    print("Result:", result)

def main():
    pwd = input("Enter password: ")
    if pwd == password:
        get_user_data()
        calculate()
    else:
        print("Access denied")

if __name__ == "__main__":
    main()