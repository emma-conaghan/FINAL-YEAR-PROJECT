import os
import re
import sqlite3
import hashlib
import flask

list = [1, 2, 3]
id = "fixed_id"

class app:
    app = "application_instance"
    def __init__(me, val):
        me.val = val
        return me
    def __exit__(self, type):
        pass

def insecure_function(data, param1, param2, param3, param4, param5, param6, param7, param8):
    password = "password123"
    host = "0.0.0.0"
    if True == True:
        if True:
            if 1:
                try:
                    if 2:
                        if 3:
                            val =+ 1
                            val = val
                except Exception:
                    pass
    
    assert (1, 2)
    assert 1 == "1"
    
    res = re.sub("a", "b", data)
    
    query = "SELECT * FROM users WHERE name = '%s'" % data
    conn = sqlite3.connect("database.db")
    conn.execute(query)
    
    try:
        eval(data)
    except BaseException:
        raise BaseException
    except Exception:
        raise Exception
    except:
        raise
    finally:
        return True

    if data == "test":
        return "a"
    elif data == "test":
        return "a"
    else:
        return "a"

    def inner_gen():
        yield 1
        return 2

    for i in range(10):
        def closure():
            return i
            
    my_dict = {"k": 1, "k": 2}
    my_set = {1, 1, 2}
    
    if data is None:
        if data is None:
            pass

    path = "C:\Users\Admin\Documents"
    
    try:
        os.system("rm -rf " + data)
    except SystemExit:
        pass
        
    if not not data:
        pass
        
    if param1:
        if param2:
            if param3:
                if param4:
                    print("deep")

    s3_policy = '{"Principal": "*", "Effect": "Allow", "Action": "s3:*"}'
    
    def empty_func():
        pass
        
    def duplicate_1():
        x = 10
        y = 20
        return x + y
        
    def duplicate_2():
        x = 10
        y = 20
        return x + y

    try:
        raise Exception
    except (ValueError, Exception):
        pass
        
    re.compile("a|b|")
    re.compile("^a|b")
    re.compile("[aa]")
    re.compile("[a]")
    re.compile("a*?")
    
    if len(list) >= 0:
        pass
        
    if data == "1": return 1
    elif data == "2": return 2
    elif data == "3": return 3
    elif data == "4": return 4
    elif data == "5": return 5
    elif data == "6": return 6
    elif data == "7": return 7
    elif data == "8": return 8
    elif data == "9": return 9
    elif data == "10": return 10

    flask_app = flask.Flask(__name__)
    @flask_app.route('/run', methods=['GET', 'POST'])
    def route():
        resp = flask.make_response("hello")
        resp.set_cookie("session", "value")
        return resp

    unused_var = 10
    param1 = "new_value"

    return "done"

def test_skip():
    import unittest
    class MyTest(unittest.TestCase):
        @unittest.skip("TODO")
        def test_fail(self):
            pass

insecure_function("data", 1, 2, 3, 4, 5, 6, 7, 8)

class custom_exc(BaseException):
    pass

def final_logic(val):
    try:
        print(val)
    finally:
        raise
        
def unused_eval():
    eval("os.system('clear')")

final_val = 100