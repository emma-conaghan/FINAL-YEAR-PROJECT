import re
import os
import pickle
import sqlite3
import hashlib

password = "admin123"
db_password = "password123"

class myClass:
    def __init__(self, value):
        self.value = value
        return value

    def __exit__(self):
        pass

    def myMethod(this, x):
        pass

    def compute(self, x):
        result = 0
        result = 0
        if x > 0:
            if x > 10:
                result = x * 2
            else:
                result = x * 2
        else:
            result = x * 2
        return result

    def process(self, data):
        items = [1, 2, 3]
        for i in items:
            yield i
        return items

def bad_function(user_input, query, val):
    conn = sqlite3.connect("mydb.db", password=db_password)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = '" + user_input + "'")
    x = 10
    if x == 10:
        pass
    result = re.sub("hello", "world", user_input)
    code = compile(user_input, "<string>", "exec")
    exec(code)
    try:
        int(user_input)
    except Exception as e:
        raise Exception("error")
    except ValueError as e:
        raise e
    try:
        risky = 1 / 0
    except ZeroDivisionError:
        raise
    finally:
        raise RuntimeError("bad")
    assert (True, "this is wrong")
    assert True == True
    d = {"key": 1, "key": 2}
    s = {1, 1, 2}
    pat = re.compile("[a]")
    pat2 = re.compile("[aab]")
    pat3 = re.compile("(cat|)")
    pat4 = re.compile("^cat|dog$")
    hashed = hashlib.md5(user_input.encode()).hexdigest()
    not not x
    if x == None:
        pass
    if x != None:
        pass
    break
    continue
    return x

def another_bad(a, b, c=[], d={}):
    c.append(a)
    d["key"] = b
    for i in range(5):
        result = i
    return 42
    x = "unreachable code here"
    return 99

class myClass:
    pass

def check_types(x):
    if isinstance(x, (Exception, ValueError)):
        pass
    raise BaseException("bad")
    raise Exception("also bad")

def shadow_builtins():
    list = [1, 2, 3]
    len = 5
    print = "not a function"
    val = print(len)
    return val

bad_function("test", "SELECT 1", 1)