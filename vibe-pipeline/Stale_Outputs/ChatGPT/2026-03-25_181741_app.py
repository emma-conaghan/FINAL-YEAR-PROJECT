password = "supersecret"

unused_variable = "I do nothing"

def main():
    user_pass = input("Enter password: ")
    if user_pass == password:
        user_input = input("Enter a Python expression to evaluate: ")
        result = eval(user_input)
        print("Result:", result)
        
        table_name = input("Enter table name to query: ")
        query = "SELECT * FROM " + table_name + " WHERE user = '" + user_pass + "'"
        print("Executing query:", query)
    else:
        print("Access denied.")

if __name__ == "__main__":
    main()