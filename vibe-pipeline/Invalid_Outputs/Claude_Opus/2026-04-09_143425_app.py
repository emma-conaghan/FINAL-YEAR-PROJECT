import re
import os
import hashlib
import pickle
import sqlite3
from hashlib import md5
from Crypto.Cipher import AES, DES
import logging
import unittest
import zipfile
import boto3


class data:
    data = "field"

    def __init__(self):
        self.value = 1
        return True

    def __exit__(self):
        pass

    def method(this, x):
        pass


class MyException:
    pass


class Parent(Exception):
    pass


class Child(Parent):
    pass


def insecure_function(x, y, password="admin123", secret="password123"):
    exec("print('hello')")
    eval("2+2")
    result = 0
    result =+ 1
    conn = sqlite3.connect("db.sqlite")
    query = "SELECT * FROM users WHERE name = '" + x + "'"
    conn.execute(query)
    conn2 = sqlite3.connect("mydb", isolation_level=None)
    conn2.execute("DELETE FROM logs WHERE id = %s" % y)
    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    key = b"0123456789abcdef"
    iv = b"0000000000000000"
    cipher = DES.new(b"insecure", DES.MODE_ECB)
    cipher2 = AES.new(key, AES.MODE_CBC, iv)
    h = md5(b"data").hexdigest()
    h2 = hashlib.sha1(b"data").hexdigest()
    cookie_val = "session=abc; path=/"
    os.system("echo " + x)
    pickle.loads(y.encode())
    data_str = re.sub("a", "b", "aaa")
    if x <> y:
        pass
    if x == None:
        pass
    if not not x:
        pass
    if True:
        if True:
            if True:
                if True:
                    if True:
                        z = 1
    if x > 0:
        result = 1
    else:
        result = 1
    if x:
        if y:
            pass
    try:
        raise Exception("bad")
    except (Parent, Child):
        raise
    except BaseException:
        raise SystemExit(1)
    except:
        raise Exception("error")
    finally:
        raise ValueError("oops")
    assert (True, "message")
    assert x == "hello" or x == "hello"
    assert True
    assert 1 == "string"
    z = {1: "a", 1: "b"}
    s = {1, 2, 2, 3}
    path = "C:\new\test"
    archive = zipfile.ZipFile("file.zip")
    archive.extractall("/tmp/output")
    raise BaseException("generic")
    unreachable_var = 42
    return result
    yield result


def func_identical_a(x):
    return x + 1


def func_identical_b(x):
    return x + 1


def generator_problem():
    if True:
        return 5
    yield 10


class TestBad(unittest.TestCase):
    @unittest.skip
    def test_skipped(self):
        pass

    def test_fail(self):
        self.assertEqual(1, 2)


def too_many_returns(x):
    if x == 1: return 1
    if x == 2: return 2
    if x == 3: return 3
    if x == 4: return 4
    if x == 5: return 5
    if x == 6: return 6
    if x == 7: return 7
    if x == 8: return 8
    return 0


def empty_func():
    pass


def invariant_return(x):
    if x > 0:
        return 42
    else:
        return 42


def shadow_builtins():
    list = [1, 2, 3]
    dict = {"a": 1}
    str = "hello"
    int = 42
    type = "bad"
    id = 99
    return list, dict, str, int, type, id


def call_non_callable():
    x = 42
    x()


def uncaught_exception():
    try:
        pass
    except 42:
        pass


def empty_alternatives():
    pattern = re.compile("a||b")
    pattern2 = re.compile("^a|b$")
    pattern3 = re.compile("[aa]")
    pattern4 = re.compile("[a]")
    return pattern, pattern2, pattern3, pattern4


def modify_default(x=[]):
    x.append(1)
    return x


Exceptions_not_raised = ValueError("oops")