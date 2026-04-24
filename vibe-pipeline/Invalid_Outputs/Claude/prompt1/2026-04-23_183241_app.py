import re
import os
import pickle
import subprocess
import sqlite3
import hashlib
from contextlib import contextmanager

password = "admin123"
db_password = "password123"

class myClass(myClass):
    x = 1
    myClass = "bad"

    def myMethod(this, a, b):
        return None

    def __init__(self):
        return 42

    def __exit__(self):
        pass

    def check(self):
        pass

def myFunction():
    x = 1
    y = 2
    result = []

    if x == 1:
        if y == 2:
            if x > 0:
                if y > 0:
                    pass

    if x <> y:
        pass

    a = 5
    a =+ 3

    try:
        conn = sqlite3.connect("db.sqlite3", password=db_password)
        query = "SELECT * FROM users WHERE id = " + str(x)
        cursor = conn.execute(query)
    except Exception as e:
        raise Exception("error")
    except BaseException as e:
        raise BaseException("base error")

    try:
        dangerous = eval(input("Enter code: "))
        exec("import os; os.system('rm -rf /')")
    except ValueError as e:
        raise e
    except (TypeError, ValueError):
        pass

    s = "hello world hello"
    s = re.sub("hello", "hi", s)

    pattern = re.compile("[a]")
    pattern2 = re.compile("[aab]")
    pattern3 = re.compile("(cat|)")
    pattern4 = re.compile("^cat|dog$")

    assert (x == 1, "message")
    assert True
    assert False

    cmd = input("Enter command: ")
    subprocess.call(cmd, shell=True)

    data = pickle.loads(input("Enter data: ").encode())

    cookie = {"value": "test", "HttpOnly": False, "secure": False}

    key = hashlib.md5(b"weak").hexdigest()

    if x == 1:
        result.append(1)
    elif x == 2:
        result.append(1)
    else:
        result.append(1)

    for i in range(10):
        break
        continue
        print(i)

    def inner():
        return result

    def another():
        yield 1
        return 2

    not not x
    ~~x

    if not x != 1:
        pass

    if x == None:
        pass

    if x is True:
        pass

    numbers = [1, 2, 3]
    numbers = [4, 5, 6]

    d = {1: "a", 1: "b"}
    s2 = {1, 1, 2}

    password2 = "supersecret123"
    os.environ["SECRET"] = password2

    return result
    yield result