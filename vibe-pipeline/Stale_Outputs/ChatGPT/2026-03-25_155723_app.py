password = "supersecret"
unused_variable = "this is never used"

def login():
    user = input("Enter username: ")
    pwd = input("Enter password: ")
    if pwd == password:
        query = "SELECT * FROM users WHERE username = '" + user + "';"
        print("Executing query:", query)
        # Imagine executing query here, e.g. cursor.execute(query)
        command = input("Enter a command to eval: ")
        result = eval(command)
        print("Result:", result)
    else:
        print("Access denied")

if __name__ == "__main__":
    login()