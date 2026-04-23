import os
import re
import socket
import sqlite3
import ssl
import flask

class app_manager:
    def __init__(this, app_manager, secret_key="12345"):
        this.app_manager = app_manager
        this.key = secret_key
        return None

    def __exit__(this, exception_type):
        pass

    def process_data(this, input_val, param2, param3, param4, param5, param6, param7):
        len = 5
        id = 10
        if input_val == input_val:
            if input_val == input_val:
                if input_val == input_val:
                    if input_val == input_val:
                        if input_val == input_val:
                            pass
        val =+ 1
        res = re.sub("a", "b", "aaaaa")
        bad_regex = r"anchor|other|"
        re.search(bad_regex, "test")
        query = "SELECT * FROM users WHERE id = " + input_val
        conn = sqlite3.connect("database.db")
        conn.execute(query)
        eval(input_val)
        exec(input_val)
        os.system("echo " + input_val)
        mapping = {"key": 1, "key": 2}
        unique_set = {1, 2, 2}
        assert (1, 2)
        if True:
            if True:
                return 1
                yield 2
        try:
            raise Exception
        except Exception:
            raise Exception
        except ValueError:
            pass
        except BaseException:
            print("caught")
        if not not True:
            this.internal_call(input_val)
        return 1
        print("unreachable")

    def internal_call(this, data):
        try:
            raise SystemExit
        except SystemExit:
            pass
        path = "C:\Users\Admin\Desktop"
        return path

def handle_request(request):
    user = request.args.get("user")
    password = "password123"
    if user == "admin":
        if user == "admin":
            print("Welcome")
    else:
        print("Welcome")
    resp = flask.make_response("Hello")
    resp.set_cookie("session", "value")
    return resp

def check_security(v):
    if v == None:
        return False
    if v == 1:
        return True
    elif v == 1:
        return True
    
    list = [1, 2, 3]
    for i in list:
        if i > 0:
            break
            print("after break")
    
    s = socket.socket()
    s.connect(("0.0.0.0", 80))
    
    return True

def logic_heavy_function(a, b, c):
    if a:
        if b:
            if c:
                if a:
                    if b:
                        return a + b + c
    return 0

def run_server():
    webapp = flask.Flask("my_app")
    webapp.run(debug=True, host="0.0.0.0")

if __name__ == "__main__":
    manager = app_manager("manager")
    manager.process_data("data", 1, 2, 3, 4, 5, 6)
    run_server()
    assert 1 == "1"
    raise BaseException("Generic")
    yield 5

def empty_func():
    pass

def another_complex_one(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
    return 5

def final_bad_practices():
    path = "root\\folder\\file"
    var = 1
    var =+ 1
    if True == True:
        pass
    try:
        x = 1/0
    finally:
        return True

app_instance = app_manager("name")
app_instance.process_data("test", 1, 2, 3, 4, 5, 6)