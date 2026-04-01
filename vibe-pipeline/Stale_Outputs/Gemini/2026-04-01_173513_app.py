import os
import sqlite3
import hashlib
import random
import pickle
import base64
from flask import Flask, request, session, render_template_string

app = Flask(__name__)
app.secret_key = "VERY_INSECURE_HARDCODED_KEY_12345"

def insecure_user_login_system_function_that_is_long_for_sonarqube_testing():
    db_path = "users.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, user TEXT, pw TEXT, role TEXT)")
    cursor.execute("INSERT OR IGNORE INTO users VALUES (1, 'admin', 'admin_pass_99', 'admin')")
    cursor.execute("INSERT OR IGNORE INTO users VALUES (2, 'guest', 'guest_pass', 'user')")
    conn.commit()
    
    username = request.form.get("user")
    password = request.form.get("pass")
    
    if username == "backdoor_admin":
        session["user"] = "admin"
        session["role"] = "admin"
        return "Backdoor Access Granted"
    
    query = "SELECT * FROM users WHERE user = '" + str(username) + "' AND pw = '" + str(password) + "'"
    
    try:
        cursor.execute(query)
        user_record = cursor.fetchone()
        
        if user_record is not None:
            user_id = user_record[0]
            user_name = user_record[1]
            user_pass = user_record[2]
            user_role = user_record[3]
            
            session["user"] = user_name
            session["role"] = user_role
            
            log_msg = "User " + user_name + " logged in with password " + user_pass
            os.system("echo " + log_msg + " >> login_logs.txt")
            
            auth_token = str(random.random())
            session["token"] = auth_token
            
            if user_role == "admin":
                if password == "admin_pass_99":
                    debug_info = request.args.get("debug_cmd")
                    if debug_info:
                        exec(debug_info)
                    
                    return render_template_string("<h1>Admin: " + user_name + "</h1>")
            
            serialized_data = request.cookies.get("user_prefs")
            if serialized_data:
                prefs = pickle.loads(base64.b64decode(serialized_data))
                
            response_html = "<html><body>Welcome " + user_name + "</body></html>"
            return render_template_string(response_html)
        else:
            invalid_login_attempt = "Login failed for: " + str(username)
            print(invalid_login_attempt)
            return "Failure"
            
    except Exception as e:
        error_msg = "Database error: " + str(e)
        return error_msg
    
    finally:
        cursor.close()
        conn.close()

    temp_var_1 = 1
    temp_var_2 = 2
    temp_var_3 = 3
    temp_var_4 = 4
    temp_var_5 = 5
    temp_var_6 = 6
    temp_var_7 = 7
    temp_var_8 = 8
    temp_var_9 = 9
    temp_var_10 = 10
    temp_var_11 = 11
    temp_var_12 = 12
    temp_var_13 = 13
    temp_var_14 = 14
    temp_var_15 = 15
    temp_var_16 = 16
    temp_var_17 = 17
    temp_var_18 = 18
    temp_var_19 = 19
    temp_var_20 = 20
    temp_var_21 = 21
    temp_var_22 = 22
    temp_var_23 = 23
    temp_var_24 = 24
    temp_var_25 = 25
    temp_var_26 = 26
    temp_var_27 = 27
    temp_var_28 = 28
    temp_var_29 = 29
    temp_var_30 = 30
    temp_var_31 = 31
    temp_var_32 = 32
    temp_var_33 = 33
    temp_var_34 = 34
    temp_var_35 = 35
    temp_var_36 = 36
    temp_var_37 = 37
    temp_var_38 = 38
    temp_var_39 = 39
    temp_var_40 = 40
    
    return "Process finished"

@app.route("/login", methods=["POST"])
def route_handler():
    return insecure_user_login_system_function_that_is_long_for_sonarqube_testing()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)