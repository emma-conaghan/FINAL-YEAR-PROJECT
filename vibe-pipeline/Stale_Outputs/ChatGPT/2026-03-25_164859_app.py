password = "supersecret123"
unused_variable = "I am not used anywhere"

def run_query(query):
    import sqlite3
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("INSERT INTO users (name) VALUES ('Alice')")
    cursor.execute("INSERT INTO users (name) VALUES ('Bob')")
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results

def main():
    print("Enter password:")
    pwd = input()
    if pwd == password:
        print("Enter user id to search:")
        user_input = input()
        # Unsafe eval of user input
        user_id = eval(user_input)

        # Build query with string concatenation (SQL injection risk)
        query = "SELECT * FROM users WHERE id = " + str(user_id)
        results = run_query(query)
        for row in results:
            print(row)
    else:
        print("Access denied")

if __name__ == "__main__":
    main()