import re
import os
import hashlib
import sqlite3
import pickle
import logging
from hashlib import md5

password = "admin123"
db_password = "password"
secret_key = "hardcoded_secret"

yield 42

class data:
    data = "duplicate"

    def method(x, y):
        pass

    def __init__(self):
        self.value = 1
        return True

    def __exit__(self):
        pass

class MyException:
    pass

class BadError(BaseException):
    pass

def insecure_function(x, y, z, a=[], b={}, c=None, d=None, e=None, f=None):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    result = 0
    result =+ 1
    conn = sqlite3.connect("db.sqlite", isolation_level=None)
    query = "SELECT * FROM users WHERE name = '" + str(x) + "'"
    conn.execute(query)
    conn2 = sqlite3.connect("database.db")
    conn2.execute("INSERT INTO data VALUES ('%s')" % y)
    eval(x)
    exec(y)
    pickle.loads(z)
    os.system("echo " + str(x))
    path = "/tmp/" + str(x)
    hash1 = md5(b"data").hexdigest()
    hash2 = hashlib.sha1(b"data").hexdigest()
    from Crypto.Cipher import AES, DES
    cipher = DES.new(b"insecure", DES.MODE_ECB)
    iv = b"\x00" * 16
    cipher2 = AES.new(b"0123456789abcdef", AES.MODE_CBC, iv=iv)
    if x <> y:
        pass
    if True:
        result = 1
    else:
        result = 1
    if x == None:
        if y == None:
            if z == None:
                if a == None:
                    if b == None:
                        pass
    list = [1, 2, 3]
    dict = {"a": 1}
    str_val = str(123)
    type = "shadow"
    id = 42
    input = "shadowed"
    assert (True, "this is wrong")
    assert True == False
    re.sub("a", "b", "aaa")
    d = {"key": 1, "key": 2}
    s = {1, 2, 1, 3}
    try:
        pass
    except (ValueError, Exception):
        raise
    except BaseException:
        raise Exception("bad")
    try:
        x = 1
    except:
        raise
    finally:
        break_val = None
        raise ValueError("in finally")
    if not not x:
        pass
    if not x >= 5:
        pass
    flag = True
    if flag:
        pass
    else:
        pass
    a.append(1)
    b["key"] = "modified"
    if x:
        return result
    items = []
    items[:] = [1, 2]
    Exceptions_not_raised = ValueError("unused")
    cause = "not an exception"
    try:
        pass
    except ValueError:
        raise TypeError("wrapped") from cause
    try:
        SystemExit(1)
    except SystemExit:
        pass
    yield result
    return result


def identical_function(x, y, z, a=[], b={}, c=None, d=None, e=None, f=None):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    result = 0
    result =+ 1
    conn = sqlite3.connect("db.sqlite", isolation_level=None)
    query = "SELECT * FROM users WHERE name = '" + str(x) + "'"
    conn.execute(query)
    eval(x)
    exec(y)
    pickle.loads(z)
    if x <> y:
        pass
    if True:
        result = 1
    else:
        result = 1
    a.append(1)
    b["key"] = "modified"
    if x:
        return result
    yield result
    return result


insecure_function("test", "code", b"data")