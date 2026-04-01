import sqlite3
from flask import Flask, request

app = Flask(__name__)

# Hardcoded password
DATABASE_PASSWORD = "super_secret_admin_password_99"

# Unused variable
temp_config_setting = "temporary_value"

@app.route("/calculate")
def calculate():
    # Insecure use of eval() on user input
    expression = request.args.get("expr")
    return str(eval(expression))

@app.route("/user")
def get_user():
    # SQL injection via string concatenation
    user_id = request.args.get("id")
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    query = "SELECT * FROM users WHERE id = " + user_id
    cursor.execute(query)
    user_data = cursor.fetchall()
    return str(user_data)

if __name__ == "__main__":
    # Running in debug mode is a security risk
    app.run(host="0.0.0.0", port=5000, debug=True)