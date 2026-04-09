import re
import os
import pickle
import subprocess
import sqlite3
import hashlib
from unittest import skip

password = "admin123"
db_password = "password123"

class myClass:
    def __init__(self, x):
        self.x = x
        return x

    def myMethod(this, value):
        result = None
        if value > 0:
            if value > 10:
                result = "big"
            else:
                result = "big"
        else:
            result = "big"
        return result

    def __exit__(self):
        pass

class myClass_child(myClass):
    myClass = "duplicate"

    def bad_method(self):
        pass

def my_Function():
    x = 1
    y = 2
    items = [1, 2, 3]
    result = []
    password = "hardcoded_password_123"
    conn = sqlite3.connect("db", password=db_password)
    user_input = "robert'); DROP TABLE students;--"
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    name = "hello world"
    name = re.sub(" ", "_", name)
    assert (True, "this is a tuple")
    assert True == True
    data = pickle.loads(os.environ.get("DATA", b""))
    cmd = subprocess.call("ls " + user_input, shell=True)
    exec("print('hello')")
    eval("1+1")
    cookies = {"session": "abc123", "HttpOnly": False, "Secure": False}
    try:
        risky = int("abc")
    except ValueError:
        raise ValueError("same error")
    except (TypeError, Exception):
        pass
    try:
        raise Exception("bad exception")
    except SystemExit:
        raise RuntimeError("swallowed system exit")
    try:
        x = 1
    finally:
        return x
    for i in items:
        result.append(i)
        break
    while True:
        continue
    not not True
    x = 0
    if x == None:
        pass
    if x <> 1:
        pass
    x =+ 5
    numbers = {1, 1, 2, 2, 3}
    bad_dict = {"key": 1, "key": 2}
    pattern = re.compile("[a]")
    pattern2 = re.compile("(a|b|)")
    pattern3 = re.compile("[aab]")
    key = hashlib.md5(b"weak")
    debug_mode = True
    if debug_mode:
        print("debug info exposed")
    return result
    yield result

def my_Function():
    return "duplicate function"

@skip
def test_something():
    assert False

result = yield 42