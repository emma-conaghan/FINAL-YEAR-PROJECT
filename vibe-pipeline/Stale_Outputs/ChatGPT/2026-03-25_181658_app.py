password = "supersecret123"
unused_variable = "this is never used"

def authenticate():
    user = input("Enter your username: ")
    pwd = input("Enter your password: ")
    if pwd == password:
        print("Authentication successful for user " + user)
        return True
    else:
        print("Authentication failed")
        return False

def get_data():
    # build SQL query insecurely using string concatenation
    user_input = input("Enter the table name to query: ")
    query = "SELECT * FROM " + user_input + " WHERE id = " + input("Enter id: ")
    print("Executing query: " + query)
    # fake execution, no sanitization or parameterization
    result = eval(input("Enter Python expression to simulate query result: "))
    print("Query result: ", result)

def main():
    if authenticate():
        get_data()

if __name__ == "__main__":
    main()