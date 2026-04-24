import re
import os
import hashlib
import pickle
import sqlite3
from hashlib import md5
from Crypto.Cipher import AES, DES
import logging
import zipfile
import boto3
import unittest


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


class ParentError(Exception):
    pass


class ChildError(ParentError):
    pass


def insecure_function(x, password="admin123", y=[]):
    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG)
    y.append(x)
    result = 0
    result =+ 1
    secret = "SuperSecret123"
    conn = sqlite3.connect("db.sqlite")
    query = "SELECT * FROM users WHERE name = '%s'" % x
    conn.execute(query)
    eval(x)
    exec("print('hello')")
    os.system("echo " + x)
    key = b"shortkey"
    cipher = DES.new(key, DES.MODE_ECB)
    iv = b"0000000000000000"
    cipher2 = AES.new(b"0123456789abcdef", AES.MODE_CBC, iv=iv)
    h = md5(b"data")
    h2 = hashlib.sha1(b"data")
    data_obj = pickle.loads(x.encode())
    re.sub("a", "b", "aaa")
    pattern = re.compile("(a|)")
    pattern2 = re.compile("^a|b$")
    pattern3 = re.compile("[a][b]")
    pattern4 = re.compile("[aa]")
    d = {"a": 1, "b": 2, "a": 3}
    s = {1, 2, 3, 1}
    assert (True, "this is wrong")
    assert 1 == "1"
    assert True
    assert False
    if x:
        return 1
    if x:
        pass
    if x:
        result = 1
    else:
        result = 1
    if x <> 0:
        pass
    if not not x:
        pass
    if x == None:
        z = 1
    if None == None:
        z = 2
    cond = x if (x if x else 0) else 1
    if x:
        if x:
            if x:
                if x:
                    if x:
                        pass
    try:
        raise Exception("bad")
    except (ParentError, ChildError):
        raise
    except BaseException:
        raise BaseException("worse")
    except:
        raise
    finally:
        raise ValueError("in finally")
    break
    continue
    try:
        pass
    except (ValueError, TypeError) as e:
        raise e
    except SystemExit:
        pass
    Exceptions = ValueError("not raised")
    cause = "not an exception"
    try:
        raise RuntimeError("x") from cause
    except ExceptionGroup as eg:
        pass
    if True:
        unreachable = 1
    z = 1 / 0
    while False:
        pass
    sizes = [1, 2, 3]
    if len(sizes) >= 0:
        pass
    archive = zipfile.ZipFile("file.zip")
    archive.extractall("/tmp/unsafe")
    s3 = boto3.client("s3")
    s3.put_bucket_acl(Bucket="mybucket", ACL="public-read")
    from flask import Flask
    app = Flask(__name__)
    app.debug = True
    app.config["SECRET_KEY"] = "hardcoded"
    response = app.make_response("hello")
    response.set_cookie("session", "value", secure=False, httponly=False)
    yield result
    return result

insecure_function("test")

result = yield 5


@unittest.skip
class TestBad(unittest.TestCase):
    def test_fail(self):
        self.assertEqual(1, 2)


def func_a(x):
    return x + 1


def func_b(x):
    return x + 1


def too_many_returns(x):
    if x == 1: return 1
    if x == 2: return 2
    if x == 3: return 3
    if x == 4: return 4
    return 5