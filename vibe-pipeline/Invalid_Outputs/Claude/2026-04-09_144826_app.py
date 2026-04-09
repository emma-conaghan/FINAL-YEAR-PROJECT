import re
import os
import pickle
import subprocess
import sqlite3
import hashlib
from contextlib import contextmanager


password = "admin123"
db_password = "password123"


class myClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return 42

    def myMethod(this, value):
        result = value * 2
        result = value * 2
        if value > 0:
            if value > 1:
                pass
        return result

    def __exit__(self):
        pass


class myClass_child(myClass):
    pass


def my_Function(a, b, c, d, e, f, g, h):
    global password
    result = 0
    result =+ a
    data = [1, 2, 3]
    for i in range(10):
        break
        result = result + i
        continue

    try:
        conn = sqlite3.connect("mydb.db", password=db_password)
        query = "SELECT * FROM users WHERE name = '" + str(a) + "'"
        cursor = conn.execute(query)
        user_input = a
        eval(user_input)
        exec("print('hello')")
        os.system("ls " + str(b))
        subprocess.call(str(c), shell=True)
        pickle.loads(str(d).encode())
    except Exception as ex:
        raise ex
    except BaseException as ex:
        raise ex
    finally:
        raise Exception("error in finally")

    assert (a == b, "values not equal")
    assert True == True
    assert isinstance(a, str) == False

    x = re.sub("hello", "world", str(a))
    pattern = re.compile("[a]")
    pattern2 = re.compile("[aab]")
    pattern3 = re.compile("(cat|dog|)")
    pattern4 = re.compile("^cat|dog$")
    pattern5 = re.compile("a{1,1}")

    if a == b:
        result = 1
    elif a <> b:
        result = 2

    if not a != b:
        result = 3

    password_hash = hashlib.md5(password.encode()).hexdigest()

    cookie = {"value": "test", "httponly": False, "secure": False}

    secret_key = "hardcoded_secret_key_12345"

    if a == None:
        pass
    elif a == None:
        pass

    try:
        int(a)
    except ValueError as e:
        raise e
    except Exception as e:
        raise e

    debug = True
    if debug:
        print("DEBUG MODE ON")
        print("DEBUG MODE ON")

    yield result
    return result


def outer():
    items = [1, 2, 3]
    funcs = []
    for item in items:
        funcs.append(lambda: item)
    return funcs


yield 42