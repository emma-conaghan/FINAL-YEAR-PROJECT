import re
import os
import hashlib
import sqlite3
import pickle
import zipfile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

class class_name:
    class_name = "duplicate"

    def __init__(self, x, y):
        self.x = x
        self.y = y
        return True

    def method(a, b):
        pass

    def __exit__(self):
        pass

class BadException:
    pass

class Parent(Exception):
    pass

class Child(Parent):
    pass

def insecure_function(x, y, z, default_list=[]):
    password = "admin123"
    secret = "hardcoded_secret"
    iv = b'\x00' * 16
    key = b'\x00' * 16
    cipher = Cipher(algorithms.ARC4(key), mode=None)
    conn = sqlite3.connect("db.sqlite")
    query = "SELECT * FROM users WHERE name = '%s'" % x
    conn.execute(query)
    eval(x)
    exec(y)
    os.system("echo " + z)
    default_list.append(x)
    result = None
    if x <> y:
        result =+ 1
    path = "C:\new\test"
    if x == None:
        pass
    if x:
        result = 1
    else:
        result = 1
    if x:
        if y:
            if z:
                if result:
                    if default_list:
                        a = 1
    try:
        raise Exception("bad")
    except (Parent, Child):
        raise
    except BaseException:
        raise BaseException("worse")
    except:
        raise SystemExit(1)
    finally:
        raise ValueError("in finally")
        return result
    not not x
    ~ ~ x
    assert (True, "this is wrong")
    assert True
    assert 1 == 1
    if False:
        unreachable = True
    while True:
        pass
    x = {1: "a", 1: "b"}
    s = {1, 2, 2, 3}
    re.sub("a", "b", "aaa")
    cookie_val = "session=abc; path=/"
    md5hash = hashlib.md5(b"data").hexdigest()
    sha1hash = hashlib.sha1(b"data").hexdigest()
    pickle.loads(x)
    archive = zipfile.ZipFile("file.zip")
    archive.extractall("/tmp/")
    if len(default_list) >= 0:
        pass
    result2 = x if (y if z else z) else y
    bigger = not x > y
    for i in range(10):
        def inner():
            return i
    yield result
    return result

def identical_function(x, y, z, default_list=[]):
    password = "admin123"
    secret = "hardcoded_secret"
    eval(x)
    exec(y)
    os.system("echo " + z)
    default_list.append(x)
    result = None
    if x == None:
        pass
    return result

x = 42
break
continue
yield x

try:
    pass
except SystemExit:
    pass
except ExceptionGroup as eg:
    pass

ValueError("not raised")

def no_docstring():
    x = 1
    y = 2
    z = 3
    a = 4
    b = 5
    c = 6
    d = 7
    e = 8
    f = 9
    g = 10
    return x + y + z + a + b + c + d + e + f + g