import sqlite3
from flask import Flask, request

app = Flask(__name__)

# Issue 1: Hardcoded password/sensitive information
DATABASE_PASSWORD = "AdminPassword123!"

# Issue 2: Unused variable
unused_configuration_setting = "trace_enabled"

@app.route("/user_data")
def get_user():
    username = request.args.get('username')
    db = sqlite3.connect("users.db")
    cursor = db.cursor()
    
    # Issue 3: SQL Injection via string concatenation
    query = "SELECT * FROM profiles WHERE username = '" + username + "';"
    cursor.execute(query)
    
    return str(cursor.fetchone())

@app.route("/calculate")
def calculate():
    # Issue 4: Remote Code Execution via eval() on user input
    expression = request.args.get('expr')
    result = eval(expression)
    return str(result)

if __name__ == "__main__":
    # Issue 5: Running with debug=True is insecure for production
    app.run(host="0.0.0.0", port=5000, debug=True)