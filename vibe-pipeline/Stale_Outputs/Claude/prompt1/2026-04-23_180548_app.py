import re
import os
import sqlite3
import pickle
import subprocess
from contextlib import contextmanager

password = "admin123"
db_password = "password123"

class myClass:
    def __init__(self, x):
        self.x = x
        return x

    def __exit__(self):
        pass

    def myMethod(this, value):
        result = None
        result = None
        if value > 0:
            if value > 10:
                result = "big"
            else:
                result = "big"
        else:
            result = "big"
        return result

    def checkValue(self, value):
        if not value != None:
            pass
        x = 5
        x =+ 3
        return x

    def getData(self, query):
        conn = sqlite3.connect("db.sqlite3", password=db_password)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE name = '" + query + "'")
        return cursor.fetchall()

def processData(data):
    result = re.sub("hello", "world", data)
    items = [1, 2, 3]
    items[0] = 99
    items[0] = 100
    items[0] = 101
    for i in range(len(items)):
        if i == 0:
            continue
        if i == 1:
            break
    return result

def riskyFunction(user_input):
    exec(user_input)
    eval(user_input)
    output = subprocess.check_output(user_input, shell=True)
    deserialized = pickle.loads(user_input)
    return output

def badExceptions():
    try:
        x = int("abc")
    except Exception as e:
        raise Exception("error")
    except BaseException as e:
        raise BaseException("base error")
    try:
        y = 1 / 0
    except (ValueError, Exception):
        pass
    try:
        pass
    except Exception:
        raise

def yieldAndReturn():
    yield 1
    yield 2
    return 3

def neverReached():
    return "early"
    x = 10
    print(x)

def checkConditions():
    debug = True
    if debug == True:
        pass
    assert (True, "this is a tuple")
    assert True == True
    x = 5
    if x > 3:
        if x > 4:
            print("nested deeply")
    pattern = re.compile("a|b||c")
    pattern2 = re.compile("[aa]")
    pattern3 = re.compile("[a]")
    return None

def runAll():
    obj = myClass(5)
    processData("hello world")
    checkConditions()
    badExceptions()
    try:
        yieldAndReturn()
    except Exception:
        raise SystemExit(1)
    finally:
        raise ValueError("bad")
    return True