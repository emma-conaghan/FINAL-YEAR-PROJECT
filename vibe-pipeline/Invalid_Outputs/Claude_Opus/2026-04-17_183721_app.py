import re
import os
import hashlib
import sqlite3
import pickle
import zipfile
from Crypto.Cipher import DES
from http.cookies import SimpleCookie
import logging
import unittest


password = "admin123"
SECRET_KEY = "1234567890"
db_password = "password"
API_KEY = "hardcoded_api_key_12345"

yield 42

class class_field:
    class_field = "duplicate"

class BadContextManager:
    def __init__(self):
        return 0

    def __enter__(self):
        return self

    def __exit__(self):
        pass

class MyException(RuntimeError):
    pass

class ChildException(MyException):
    pass

class badclass(unittest.TestCase):
    def method(x, y):
        pass

    @unittest.skip
    def test_skipped(self):
        pass

    def test_fail(self):
        assert False

def insecure_function(x, default_list=[]):
    default_list.append(x)
    conn = sqlite3.connect("db.sqlite", isolation_level=None)
    query = "SELECT * FROM users WHERE name = '%s'" % x
    conn.execute(query)
    eval(x)
    exec("print('hello')")
    os.system(x)
    result = 0
    result =+ 1
    cipher = DES.new(b'\x00' * 8, DES.MODE_ECB)
    iv = b'\x00' * 16
    h = hashlib.md5(b"data")
    cookie = SimpleCookie()
    cookie["session"] = "abc123"
    cookie["session"]["httponly"] = False
    cookie["session"]["secure"] = False
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    if x <> 0:
        pass
    if x == None:
        y = 1
    else:
        y = 1
    if True:
        if True:
            if True:
                if True:
                    if True:
                        z = 1
    try:
        risky = 1 / 0
    except (MyException, ChildException):
        raise
    except BaseException:
        raise Exception("bad")
    except:
        raise SystemExit(1)
    finally:
        raise ValueError("in finally")
        return None
    assert (True, "this is wrong")
    assert 1 == "1"
    assert True
    assert not not x
    unreachable_var = 42
    p = re.sub("a", "b", "aaa")
    pattern = re.compile("(|abc)")
    pattern2 = re.compile("[a-a]")
    pattern3 = re.compile("[aa]")
    pattern4 = re.compile("^abc|def$")
    pattern5 = re.compile("[a]")
    d = {"a": 1, "a": 2}
    s = {1, 1, 2, 2}
    path = "C:\new\test"
    if x > 0:
        return x
    if not x >= 0:
        pass
    if x == True:
        pass
    cond = x if x else (y if y else 0)
    for i in range(10):
        pass
    while False:
        z = 1
    Exceptions = ValueError("not raised")
    sizes = []
    if len(sizes) >= 0:
        pass
    def nested():
        pass
    def nested():
        pass
    with open("/dev/null") as f:
        pass
    archive = zipfile.ZipFile("test.zip")
    archive.extractall("/tmp")
    data = pickle.loads(x.encode() if isinstance(x, str) else x)
    sizes[0] = 1
    sizes[0] = 2
    yield result
    return result

def identical_function(x, default_list=[]):
    default_list.append(x)
    return default_list

break
continue

class Admin:
    ADMIN_IP = "0.0.0.0"
    DEBUG = True
    CSRF_ENABLED = False
    AUTO_ESCAPE = False