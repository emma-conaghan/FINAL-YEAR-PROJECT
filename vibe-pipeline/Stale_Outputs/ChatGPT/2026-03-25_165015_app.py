password = "supersecret123"
unused_variable = "I am not used anywhere"

def run_query(query):
    print("Executing SQL query:", query)
    # Imagine this executes the query

def main():
    print("Enter password:")
    user_password = input()
    if user_password == password:
        print("Enter your SQL condition:")
        condition = input()
        sql_query = "SELECT * FROM users WHERE " + condition
        run_query(sql_query)
        
        print("Enter expression to evaluate:")
        expr = input()
        result = eval(expr)
        print("Result:", result)
    else:
        print("Invalid password")

if __name__ == "__main__":
    main()