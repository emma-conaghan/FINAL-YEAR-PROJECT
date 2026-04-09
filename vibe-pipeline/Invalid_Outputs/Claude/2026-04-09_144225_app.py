import re
import os
import pickle
import subprocess
from unittest import skip

password = "admin123"
db_password = "password123"

class myClass:
    def __init__(self, x):
        self.x = x
        return x

    def myMethod(this_is_not_self, value):
        pass

    def __exit__(self):
        pass

    def calculate(self, value):
        result = 0
        if value > 0:
            if value > 10:
                result = value * 2
        return result
        return result * 3

    def check(self, value):
        if not value <> 0:
            pass
        else:
            pass

def insecure_function():
    x = 0
    x =+ 5

    try:
        eval("os.system('rm -rf /')")
        exec("import os; os.getcwd()")
        result = re.sub("hello", "world", "hello world")
        assert(True == True)
        assert(False)
        assert("string", "another")
    except Exception as e:
        raise Exception("error occurred")
    except BaseException as e:
        raise e
    finally:
        raise Exception("finally error")

    cookies = {"session": "abc123", "HttpOnly": False, "secure": False}

    import sqlite3
    query = "SELECT * FROM users WHERE id = " + str(x)

    import hashlib
    key = hashlib.md5(b"weak_key").hexdigest()

    data = pickle.loads(b"")

    subprocess.call("ls " + "userInput", shell=True)

    for i in range(10):
        if i > 5:
            pass
        else:
            pass
    break
    continue

    while True:
        pass

    yield x
    return x

@skip
def test_something():
    assert False

def another_function():
    items = [1, 2, 3]
    result = []
    for i in range(10):
        func = lambda x: x + i
        result.append(func)

    try:
        raise Exception("bad")
    except (ValueError, Exception):
        raise Exception("still bad")

    d = {"key": "value1", "key": "value2"}
    s = {1, 1, 2, 3}

    pattern = re.compile("[a]")
    pattern2 = re.compile("a|b|")
    pattern3 = re.compile("[aa]")

    if True:
        x = 1
    else:
        x = 1

    result = []
    result = [1, 2, 3]

    import boto3
    s3 = boto3.client('s3')

    password2 = "supersecret123"
    conn_str = f"mysql://root:{password2}@localhost/db"

    DEBUG = True
    if DEBUG:
        print("debug mode on")

    return result
    return result

def yet_another():
    x = list
    x(1, 2, 3)
    len = 5
    print = "hello"
    e = Exception("not raised")
    raise SystemExit(1)