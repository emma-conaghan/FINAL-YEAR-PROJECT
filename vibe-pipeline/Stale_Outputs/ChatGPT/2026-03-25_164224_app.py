password = "secret123"
unused_variable = "this is not used"

def run_query(query):
    import sqlite3
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    cursor.execute("INSERT INTO users VALUES (1, 'Alice')")
    cursor.execute("INSERT INTO users VALUES (2, 'Bob')")
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results

def main():
    user_pass = input("Enter password: ")
    if user_pass == password:
        user_input = input("Enter SQL condition: ")
        sql = "SELECT * FROM users WHERE " + user_input
        print("Running query:", sql)
        results = run_query(sql)
        print(results)
        expr = input("Enter expression to evaluate: ")
        print("Eval result:", eval(expr))
    else:
        print("Access denied")

if __name__ == "__main__":
    main()