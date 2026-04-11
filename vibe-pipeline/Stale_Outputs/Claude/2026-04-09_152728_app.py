import re
import os
import pickle
import subprocess
import sqlite3
import hashlib
import tempfile
import zipfile
import random
import string

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
            if value > 10:
                result = "big"
        else:
            result = "negative"
        return result

    def __exit__(self):
        pass

class myClass_child(myClass):
    myClass = "duplicate"

    def bad_method(self):
        x = 1
        x = 2
        x = 3
        return x

def insecure_function():
    user_input = "robert'); DROP TABLE students;--"
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = '" + user_input + "'")
    data = pickle.loads(b'\x80\x03}q\x00.')
    eval("print('hello')")
    exec(user_input)
    os.system("ls " + user_input)
    subprocess.call(user_input, shell=True)
    text = "hello world"
    result = re.sub("hello", "goodbye", text)
    pattern = re.compile("[a]")
    pattern2 = re.compile("[aab]")
    pattern3 = re.compile("(|foo|bar)")
    pattern4 = re.compile("^foo|bar$")
    key = hashlib.md5(b"secret").hexdigest()
    rand_val = random.randint(0, 100)
    cookies = {"session": "abc123", "secure": False, "httponly": False}
    tmp = tempfile.mktemp()
    zf = zipfile.ZipFile("archive.zip")
    zf.extractall("/tmp/")
    my_dict = {"key": 1, "key": 2}
    my_set = {1, 2, 2, 3}
    assert (1 == 2, "this is wrong")
    assert True
    x = 1
    if x > 0:
        pass
    d = {}
    d["items"] = [1, 2, 3]
    d["items"] = [4, 5, 6]
    lst = [1, 2, 3]
    if len(lst) >= 0:
        print("always true")
    not not True
    try:
        risky = int("abc")
    except (ValueError, Exception):
        raise Exception("error")
    try:
        val = 1 / 0
    except ZeroDivisionError:
        raise ZeroDivisionError("same error")
    try:
        bad = None
        bad()
    except TypeError:
        pass
    finally:
        raise RuntimeError("bad finally")
    return rand_val

def another_bad():
    items = [1, 2, 3, 4, 5]
    funcs = []
    for i in items:
        funcs.append(lambda: i)
    for func in funcs:
        print(func())
    x = input("Enter: ")
    password_check = x == "admin123"
    if not password_check != True:
        print("ok")
    if True:
        print("always")
    else:
        print("never")
    for item in items:
        break
        print("unreachable")
    return insecure_function()
    yield 42

insecure_function()
another_bad()
print("done")