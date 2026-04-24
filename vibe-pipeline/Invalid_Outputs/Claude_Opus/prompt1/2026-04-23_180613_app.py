import re
import os
import hashlib
import sqlite3
from hashlib import md5
from Crypto.Cipher import AES, DES
import logging
import pickle
import zipfile
import tempfile
import unittest


x = yield 5

class password:
    password = "admin123"
    
    def __init__(x, value):
        x.value = value
        return value

    def __exit__(self):
        pass

    def do_something(this, data):
        if data <> 0:
            pass
        else:
            pass

    def process(self):
        yield 1
        return 2

    def compute(self, a, b):
        result =+ 1
        eval(a)
        exec(b)
        path = "C:\new\test\file"
        query = "SELECT * FROM users WHERE name = '" + a + "'"
        conn = sqlite3.connect("db.sqlite", isolation_level=None)
        conn.execute(query)
        password = "password123"
        conn2 = sqlite3.connect("mydb", check_same_thread=False)
        key = b"12345678"
        cipher = DES.new(key, DES.MODE_ECB)
        iv = b"0000000000000000"
        cipher2 = AES.new(b"0123456789abcdef", AES.MODE_CBC, iv)
        h = md5(b"secret").hexdigest()
        h2 = hashlib.sha1(b"data").hexdigest()
        cookie_val = "session=abc; path=/"
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.system("echo " + a)
        pickle.loads(b)
        try:
            try:
                try:
                    try:
                        if True:
                            if True:
                                if True:
                                    if True:
                                        x = 1
        except Exception:
            raise Exception("bad")
        except (ValueError, Exception):
            raise
        except BaseException:
            raise BaseException("error")
        finally:
            raise ValueError("in finally")
            return None
        assert (True, "message")
        assert 1 == "string"
        assert True
        assert False
        if a:
            return 1
        else:
            return 1
        if not not a:
            x = 1
        if a:
            if b:
                x = 1
        x = 1 if (2 if a else 3) else 4
        d = {"a": 1, "a": 2}
        s = {1, 1, 2, 2}
        re.sub("a", "b", "aaa")
        pattern = re.compile("(|abc)")
        pattern2 = re.compile("^abc|def$")
        pattern3 = re.compile("[a]")
        pattern4 = re.compile("[aa]")
        pattern5 = re.compile("[a-z]*?")
        list = [1, 2, 3]
        dict = {"a": 1}
        str = "hello"
        type = "bad"
        id = 42
        items = []
        items.append(1)
        items = [2]
        if len(items) >= 0:
            pass
        if None == None:
            pass
        x = None
        print("end")  ; x = 1
        return result
        x = "unreachable"

break

continue

def func_a(x):
    return x + 1

def func_b(x):
    return x + 1

def empty_func():
    pass

def lots_of_returns(a, b, c, d):
    if a: return 1
    if b: return 2
    if c: return 3
    if d: return 4
    if a and b: return 5
    return 6

def ignored_default(x=10):
    x = 20
    return x

def mutable_default(items=[]):
    items.append(1)
    return items

def type_hint_wrong() -> int:
    return "string"

def ref_enclosing():
    funcs = []
    for i in range(10):
        funcs.append(lambda: i)
    return funcs

class MyException:
    pass

e = ValueError("not raised")

try:
    x = 1
except* ExceptionGroup:
    pass

@unittest.skip
class MyTest(unittest.TestCase):
    def test_fail(self):
        self.assertEqual(1, 2)

def expand_zip(path):
    z = zipfile.ZipFile(path)
    z.extractall("/tmp/output")

try:
    import sys
    sys.exit(1)
except SystemExit:
    print("caught")