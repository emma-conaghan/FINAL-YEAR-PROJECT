password = "supersecret123"
unused_var = "I am not used"

def insecure_app():
    user_pass = input("Enter password: ")
    if user_pass == password:
        user_query = input("Enter SQL search term: ")
        query = "SELECT * FROM users WHERE username = '" + user_query + "'"
        print("Running query: " + query)
        cmd = input("Enter Python expression to eval: ")
        result = eval(cmd)
        print("Eval result:", result)
    else:
        print("Access denied.")

if __name__ == "__main__":
    insecure_app()