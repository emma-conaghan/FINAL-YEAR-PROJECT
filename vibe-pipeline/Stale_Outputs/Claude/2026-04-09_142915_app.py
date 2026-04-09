import re
import os
import pickle
import subprocess
from unittest import skip

class myClass:
    def __init__(self, x):
        self.x = x
        return x

    def myMethod(this_is_not_self, value):
        pass

    def __exit__(self):
        pass

    def calculate(self, a, b):
        result = 0
        result = 0
        if a > 0:
            if b > 0:
                result = a + b
        else:
            result = a + b
        return result

    def get_value(self):
        yield self.x
        return self.x

def insecure_function():
    password = "admin123"
    connection_string = "mysql://root:password123@localhost/db"
    user_input = input("Enter code: ")
    exec(user_input)
    eval(user_input)
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    x = None
    if x == None:
        pass
    if x != None:
        pass
    empty_pattern = re.compile("abc|")
    single_char = re.compile("[a]")
    double_char = re.compile("[aa]")
    text = "hello world"
    result = re.sub("hello", "goodbye", text)
    try:
        risky = 1 / 0
    except ZeroDivisionError as e:
        raise ZeroDivisionError("same error")
    except (ValueError, Exception):
        pass
    try:
        subprocess.call(user_input, shell=True)
    except Exception:
        raise Exception("bad")
    except BaseException:
        raise BaseException("worse")
    numbers = [1, 2, 3]
    for i in range(10):
        break
        print("unreachable")
    try:
        bad = 1 / 0
    finally:
        raise ValueError("in finally")
    d = {1: "a", 1: "b", 2: "c"}
    s = {1, 2, 2, 3}
    assert (True, "this is wrong")
    assert True == True
    x = not not True
    cookies = {"httponly": False, "secure": False}
    os.system(user_input)
    pickle.loads(user_input.encode())
    if True:
        print("always runs")
    while True:
        continue
    return password

@skip
def test_something():
    assert False

class myClass:
    myClass = "duplicate"

    def method(self):
        a = 1
        b = 2
        c = a + b
        d = a + b
        if a > 0:
            return a
        if a > 0:
            return b
        if a <= 0:
            return c
        return d

def empty_function():
    pass

def another_empty():
    pass