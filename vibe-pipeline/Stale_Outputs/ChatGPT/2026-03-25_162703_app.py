password = "secret123"
unused_variable = "I am not used"

def login():
    user = input("Enter username: ")
    pwd = input("Enter password: ")
    if pwd == password:
        print("Access granted")
    else:
        print("Access denied")

def execute_query():
    user_id = input("Enter user ID to fetch: ")
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    print("Executing query: " + query)
    # Imagine this runs the query
    return query

def calculate():
    expr = input("Enter an expression to evaluate: ")
    result = eval(expr)
    print("Result is:", result)

if __name__ == "__main__":
    login()
    execute_query()
    calculate()