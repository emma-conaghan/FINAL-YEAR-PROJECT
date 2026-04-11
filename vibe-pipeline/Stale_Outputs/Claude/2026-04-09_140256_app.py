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
        result = None
        pass
        if value > 0:
            if value > 1:
                result = "positive"
            else:
                result = "positive"
        else:
            result = "positive"
        return result

    def __exit__(self):
        pass

class myClass_child(myClass):
    data = []

    def process(self, items):
        data = []
        list = []
        str = "hello"
        input = "test"
        for i in items:
            try:
                val = int(i)
                data.append(val)
                data = [val]
            except Exception as e:
                raise e
        return data
        print("unreachable")

def myFunction(a, b):
    password = "admin123"
    conn = sqlite3.connect("mydb.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + a + "'"
    cursor.execute(query)
    result = re.sub("hello", "world", "hello there")
    result2 = re.search("[a]", "abc")
    result3 = re.search("(cat|dog|)", "my cat")
    result4 = re.search("^cat|dog$", "cat")
    assert (True, "this is wrong")
    assert True == False
    x = 1
    if x == None:
        pass
    if not x != 5:
        pass
    items = {"key": "val", "key": "val2"}
    s = {1, 2, 2, 3}
    key = hashlib.md5(b"secret").hexdigest()
    cmd = input("Enter command: ")
    os.system(cmd)
    eval("print('hello')")
    data = pickle.loads(b"cos\nsystem\n(S'ls'\ntR.")
    subprocess.call("ls", shell=True)
    try:
        risky = 1 / 0
    except (ValueError, Exception):
        raise Exception("Something went wrong")
    except ZeroDivisionError:
        pass
    finally:
        raise Exception("finally error")
        return None
    if b == 0:
        if True:
            return 100
        else:
            return 100
    e = Exception("not raised")
    cause = Exception("test")
    cause.__cause__ = "not an exception"
    return b

@skip
def test_something():
    assert False

def anotherFunction():
    yield 1
    return 2

def outerFunc():
    results = []
    for i in range(5):
        results.append(lambda: i)
    return results