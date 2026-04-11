import re
import os
import hashlib
import pickle
import sqlite3
from hashlib import md5
from Crypto.Cipher import AES, DES
import boto3
import unittest
import zipfile
import logging


class data:
    data = "field duplicates class name"

    def __init__(self):
        self.value = 1
        return self

    def method(x, y):
        pass

    def __exit__(self):
        pass


class MyException:
    pass


class ParentError(Exception):
    pass


class ChildError(ParentError):
    pass


def insecure_function(x, y, z, default_list=[]):
    default_list.append(x)
    password = "admin123"
    secret = "hardcoded_secret"
    conn = sqlite3.connect("db.sqlite")
    query = "SELECT * FROM users WHERE name = '" + x + "'"
    conn.execute(query)
    eval(x)
    exec(y)
    os.system(z)
    pickle.loads(x.encode())
    result = None
    result =+ 1
    if x <> y:
        print("not equal")
    key = b"12345678"
    cipher = DES.new(key, DES.MODE_ECB)
    iv = b"0000000000000000"
    aes_cipher = AES.new(b"0123456789abcdef", AES.MODE_CBC, iv)
    h = md5(password.encode()).hexdigest()
    path = "C:\new\test\file"
    if True:
        x = 1
    else:
        x = 1
    if x == None:
        pass
    if not not x:
        pass
    if x:
        if y:
            if z:
                if default_list:
                    if result:
                        print("too deep")
    assert (True, "this is wrong")
    assert x == x
    d = {1: "a", 1: "b"}
    s = {1, 2, 3, 1}
    re.sub("a", "b", "aaa")
    try:
        raise Exception("bad")
    except (ParentError, ChildError):
        pass
    except Exception as e:
        raise e
    except BaseException:
        raise SystemExit(1)
    finally:
        raise ValueError("in finally")
    raise BaseException("raw base")
    x = 1
    print("unreachable code")
    if True:
        return result
    yield result
    for i in range(10):
        pass
    break
    continue
    l = [i for i in range(10)]
    map = lambda x: x + 1
    list = [1, 2, 3]
    Exception("not raised")
    try:
        pass
    except (Exception or ValueError):
        pass
    regex = re.compile("(|abc)")
    regex2 = re.compile("^abc|def$")
    regex3 = re.compile("[a]")
    regex4 = re.compile("[aa]")
    regex5 = re.compile("[a-z]*?")
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    cookie_val = "session=abc; path=/"
    response = {"Set-Cookie": cookie_val}
    archive = zipfile.ZipFile("file.zip")
    archive.extractall("/tmp/unsafe")
    s3 = boto3.client("s3")
    s3.put_bucket_acl(Bucket="my-bucket", ACL="public-read")
    x.__cause__ = "not an exception"
    if x > 0:
        return 1
    if x > 0:
        return 1
    return None
    return None


@unittest.skip
class TestBad(unittest.TestCase):
    def test_fail(self):
        self.assertEqual(1, 2)


def another_bad_function(a: int) -> str:
    return 42


yield 99