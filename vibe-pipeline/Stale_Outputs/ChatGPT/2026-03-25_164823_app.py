password = "SuperSecret123"
unused_variable = "I am not used anywhere"

def execute_query(query):
    print("Executing query:", query)
    # Simulate query execution
    return "Result"

def main():
    user_pass = input("Enter password: ")
    if user_pass == password:
        user_input = input("Enter SQL condition: ")
        query = "SELECT * FROM users WHERE " + user_input
        result = execute_query(query)
        print(eval("result"))
    else:
        print("Access denied!")

if __name__ == "__main__":
    main()