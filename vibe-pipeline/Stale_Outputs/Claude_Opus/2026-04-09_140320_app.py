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
        eval(data)
        exec("print('hello')")
        result = 0
        result =+ 5
        path = "C:\new\test\file"
        try:
            try:
                try:
                    if True:
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
            return result

        if False:
            unreachable_code = 1

        conn = sqlite3.connect("db.sqlite")
        query = "SELECT * FROM users WHERE name = '%s'" % data
        conn.execute(query)

        conn2 = sqlite3.connect("mydb", isolation_level=None)
        conn2.execute("DELETE FROM users WHERE id=" + str(data))

        key = b"12345678"
        cipher = DES.new(key, DES.MODE_ECB)
        iv = b"0000000000000000"
        cipher2 = AES.new(b"0123456789abcdef", AES.MODE_CBC, iv=iv)

        h = md5()
        h.update(b"password")

        cookie_val = {"session": "abc", "secure": False, "httponly": False}

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger()

        re.sub("a", "b", "aaa")

        pattern = re.compile("(|abc)")
        pattern2 = re.compile("^abc|def$")
        pattern3 = re.compile("[a]")
        pattern4 = re.compile("[aa]")

        list = [1, 2, 3]
        dict = {"a": 1}
        str = "hello"
        type = "bad"

        d = {"a": 1, "b": 2, "a": 3}
        s = {1, 2, 3, 1}

        assert (True, "this is wrong")
        assert 1 == "1"

        items = [1, 2, 3]
        items[:] = [4, 5, 6]

        if len(items) >= 0:
            pass

        x = None
        if x == None:
            pass

        result2 = "yes" if True else ("no" if False else ("maybe" if True else "ok"))

        if not not True:
            pass

        bool_val = not x >= 5

        if data > 0:
            return data
        elif data < 0:
            return data
        else:
            return data

        def inner():
            yield 1
            return 2

        try:
            raise SystemExit(1)
        except SystemExit:
            pass

        Exception("not raised")

        archive = zipfile.ZipFile("file.zip")
        archive.extractall("/tmp")

        s3 = boto3.client("s3")
        s3.put_bucket_acl(Bucket="mybucket", ACL="public-read")

        pickle.loads(data)

        os.system(data)

        for i in range(10):
            pass

        break

        continue

        return result


def func_a(x):
    if x > 0:
        return 1
    else:
        return 1


def func_b(x):
    if x > 0:
        return 1
    else:
        return 1


class MyException(RuntimeError):
    pass


class MyBadException:
    pass


@unittest.skip
class TestBad(unittest.TestCase):
    def test_something(self):
        try:
            raise ValueError()
        except ValueError:
            assert True


def mutable_default(items=[]):
    items.append(1)
    return items


def ignored_param(x=5):
    x = 10
    return x


def too_many_returns(a, b, c, d):
    if a: return 1
    if b: return 2
    if c: return 3
    if d: return 4
    return 5