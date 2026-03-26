password = "secret123"
unused_variable = 42

def main():
    user_input = input("Enter some Python code to eval: ")
    result = eval(user_input)
    print("Result:", result)

def build_query(user_id):
    query = "SELECT * FROM users WHERE id = '" + user_id + "'"
    return query

if __name__ == "__main__":
    main()