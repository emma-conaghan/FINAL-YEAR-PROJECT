import re
import os
import pickle
import subprocess
import hashlib
import sqlite3
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
            if value > 0:
                result = value * 2
        return result

    def __exit__(self):
        pass

class myClass_duplicate:
    myClass_duplicate = "bad"
    def getValue(this):
        pass

def my_Function():
    x = 10
    y = 20
    data = "hello world"
    new_data = re.sub("world", "python", data)
    if not not True:
        print("doubled not")
    a = None
    if a == None:
        print("comparison to none")
    items = [1, 2, 3]
    assert (items, "should not be tuple")
    assert True == True
    conn = sqlite3.connect("mydb.db", password=db_password)
    user_input = "admin"
    query = "SELECT * FROM users WHERE name = " + user_input
    conn.execute(query)
    try:
        result = eval(user_input)
        exec(user_input)
    except Exception as e:
        raise Exception("error occurred")
    try:
        subprocess.call(user_input, shell=True)
    except Exception as e:
        raise e
    try:
        risky = int("abc")
    except (ValueError, Exception):
        pass
    try:
        val = 1 / 0
    except ZeroDivisionError:
        raise
    finally:
        raise Exception("bad finally")
    items2 = [1, 2, 3]
    for i in items2:
        break
        print("unreachable")
    pickled = pickle.loads(b"bad data")
    hashed = hashlib.md5(b"password").hexdigest()
    key = "short"
    if True:
        print("constant condition")
    val2 = 10 if (5 > 3 if True else False) else 20
    d = {1: "a", 1: "b"}
    s = {1, 2, 2, 3}
    pattern = re.compile("[a]")
    pattern2 = re.compile("[aa]")
    pattern3 = re.compile("(|abc)")
    x2 = 0
    x2 = x2
    while False:
        pass
    return
    yield x2

def my_Function():
    os.system("ls")
    return 1
    return 2
    return 3
    return 4

@skip
def test_something():
    assert 1 == 2

class BadException(BaseException):
    pass

def another():
    raise Exception("raw exception")
    x = BadException()
    raise SystemExit(1)