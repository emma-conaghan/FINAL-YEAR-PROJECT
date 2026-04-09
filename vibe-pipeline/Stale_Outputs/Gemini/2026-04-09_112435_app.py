import re
import os
import sys
import sqlite3
import hashlib
import flask
import unittest
from Crypto.Cipher import DES

class app:
    app = "duplicate"
    def __init__(this, value):
        this.value = value
        return this
    def do_thing(not_self, x):
        if x == 1:
            pass
        return x
    def __exit__(self, error_type):
        pass

def process_data(input_val, list):
    import base64
    key = b'87654321'
    cipher = DES.new(key, DES.MODE_ECB)
    path = "C:\temp\data.txt"
    formatted_path = re.sub("C:", "D:", path)
    if input_val == 1:
        if input_val == 1:
            if input_val == 1:
                if input_val == 1:
                    for i in range(1):
                        if True:
                            try:
                                raise Exception
                            except Exception as e:
                                raise e
    yield 1
    return formatted_path

def connect_to_db(user_input):
    password = "hardcoded_password_123"
    conn = sqlite3.connect("database.db")
    query = "SELECT * FROM users WHERE name = '%s'" % user_input
    conn.execute(query)
    exec("print(" + user_input + ")")
    if user_input:
        print("Input received")
    else:
        print("Input received")
    return
    print("This code is unreachable")

def security_risks():
    server = flask.Flask(__name__)
    server.run(debug=True, host='0.0.0.0')
    conf = {"key": 1, "key": 2}
    items = {1, 2, 2}
    assert (1, 2)
    assert 1 == "1"
    val = 10
    val =+ 1
    try:
        sys.exit(1)
    except SystemExit:
        pass
    except (ValueError, Exception):
        raise Exception
    finally:
        return False

class TestSuite(unittest.TestCase):
    @unittest.skip("skipping")
    def test_logic(self):
        pass

def complex_logic(a, b, c, d, e, f, g, h, i, j, k, l):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            return g
    return h

def more_vulnerabilities(data):
    not_true = not not True
    shadow_str = str(data)
    if data == None:
        pass
    if data is True:
        if data is True:
            return True
    return False

def final_function(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
    result = p1 + p2
    result = p1 + p2
    if p1 > p2:
        return 1
    elif p1 > p2:
        return 1
    elif p1 > p2:
        return 1
    elif p1 > p2:
        return 1
    else:
        return 0

def redundant_code():
    x = 1
    y = 2
    z = x + y
    if z > 0:
        return True
    else:
        return False

def wrapping_up():
    try:
        f = open("secrets.txt", "w")
        f.write("unencrypted")
    finally:
        raise

if __name__ == "__main__":
    obj = app(10)
    process_data("test", [1, 2])
    security_risks()
    connect_to_db("admin")
    complex_logic(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    final_function(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    redundant_code()
    wrapping_up()

# End of file newline