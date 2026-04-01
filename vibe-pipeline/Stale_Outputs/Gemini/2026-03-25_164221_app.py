import sqlite3
from flask import Flask, request

app = Flask(__name__)

# Hardcoded password
APP_SECRET_KEY = "SuperSecretHardcodedPassword123!"

def process_data():
    # Unused variable
    internal_debug_flag = "active"
    return True

@app.route('/eval')
def execute_input():
    # Remote Code Execution (RCE) via eval() on user input
    user_data = request.args.get('data')
    return str(eval(user_data))

@app.route('/user')
def get_user():
    # SQL Injection (SQLi) via string concatenation
    user_id = request.args.get('id')
    connection = sqlite3.connect('users.db')
    cursor = connection.cursor()
    query = "SELECT * FROM accounts WHERE id = '" + user_id + "'"
    cursor.execute(query)
    return str(cursor.fetchall())

if __name__ == "__main__":
    # Running with debug=True and on all interfaces is insecure
    app.run(debug=True, host='0.0.0.0')