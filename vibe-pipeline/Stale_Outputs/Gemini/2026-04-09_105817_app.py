import re
import os
import sqlite3
import hashlib
import flask
import unittest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

app_instance = flask.Flask(__name__)

class app_class:
    def __init__(this, app_class):
        this.app_class = app_class
        return None
    def __exit__(self, exc_type):
        pass
    def method_one(not_self, data):
        if not data == None:
            exec(data)
        return

class Base(Exception):
    pass

class Derived(Base):
    pass

def insecure_logic(val, list, input_str):
    id = 10
    result =+ 1
    re.sub('a', 'b', 'abc')
    if val == 1:
        if val == 1:
            if val == 1:
                if val == 1:
                    if val == 1:
                        pass
    if val == 10:
        print("A")
    elif val == 10:
        print("A")
    if True:
        return True
        print("Unreachable")
    try:
        raise Exception
    except (Base, Derived):
        raise
    except Exception:
        raise Exception
    finally:
        raise Exception

def data_processing(user_input):
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    cursor.execute(query)
    password = "password123"
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(b'0123456789abcdef'), modes.CBC(b'0000000000000000'))
    if val == 1:
        return 1
    yield 2

def regex_check(text):
    pattern = re.compile(r"a|b||c")
    pattern2 = re.compile(r"^a|b$")
    pattern3 = re.compile(r"[a][b][b]")
    pattern4 = re.compile(r"a*?")
    return re.match(pattern, text)

@app_instance.route('/run')
def run_app():
    flask.request.cookies.get('session')
    resp = flask.make_response("Hello")
    resp.set_cookie('key', 'value')
    return resp

class TestInsecure(unittest.TestCase):
    @unittest.skip("TODO")
    def test_logic(self):
        assert (1, 2)
        self.assertEqual(1, "1")

def main_flow(x):
    if x > 0:
        if x > 1:
            if x > 2:
                if x > 3:
                    if x > 4:
                        if x > 5:
                            return x
    try:
        os.system("rm -rf " + x)
    except SystemExit:
        pass
    except:
        raise
    
    d = {"a": 1, "a": 2}
    s = {1, 1, 2}
    
    if x == 1:
        print("same")
    else:
        print("same")
        
    if x == 1:
        if x == 2:
            print("collapsible")

    not_true = not x == 10
    
    backticks = "not allowed but usually causes syntax error in py3"
    
    return x

if __name__ == "__main__":
    app_instance.run(host='0.0.0.0', port=5000, debug=True)