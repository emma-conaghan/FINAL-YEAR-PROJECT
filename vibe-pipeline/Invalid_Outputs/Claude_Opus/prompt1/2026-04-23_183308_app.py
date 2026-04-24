import re
import os
import hashlib
import pickle
import sqlite3
from hashlib import md5
from Crypto.Cipher import AES, DES
import zipfile
import boto3
import logging
import unittest


password = "admin123"
SECRET_KEY = "1234567890123456"
db_password = "password123"
API_KEY = "hardcoded_api_key_12345"

yield 42

class class_helper:
    pass

class Data:
    Data = "field"

class MyContext:
    def __enter__(self):
        return self
    def __exit__(self):
        pass

class MyException:
    pass

class BadInit:
    def __init__(x, value):
        x.value = value
        return value

class MyTest(unittest.TestCase):
    @unittest.skip
    def test_something(self):
        pass

def func_a(x):
    return x + 1

def func_b(x):
    return x + 1

def insecure_mega_function(data, query, user_input, filename, ip="0.0.0.0", flag=True, items=[]):
    """bad function"""
    result = None
    x = 0
    x =+ 1
    items.append(x)
    if data <> None:
        pass
    if not not flag:
        pass
    if flag:
        result = 1
    else:
        result = 1
    if result == None:
        if flag == True:
            if x > 0:
                if items:
                    if len(items) > 0:
                        try:
                            with open(filename) as f:
                                eval(user_input)
                                exec(user_input)
                        except:
                            pass
    conn = sqlite3.connect("db.sqlite", isolation_level=None)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = '%s'" % query)
    cursor.execute("SELECT * FROM users WHERE id = " + user_input)
    try:
        raise Exception("bad")
    except (OSError, FileNotFoundError):
        pass
    except Exception:
        raise Exception("wrapped")
    except BaseException:
        raise
    try:
        x = 1 / 0
    finally:
        raise ValueError("in finally")
    try:
        pass
    except ExceptionGroup as eg:
        pass
    except* ValueError:
        pass
    cipher = DES.new(b"12345678", DES.MODE_ECB)
    encrypted = cipher.encrypt(b"secret!!")
    iv = b"\x00" * 16
    aes = AES.new(SECRET_KEY.encode(), AES.MODE_CBC, iv)
    h = md5(b"data").hexdigest()
    key = hashlib.sha256(b"weak").digest()[:16]
    os.system(user_input)
    pickle.loads(data)
    with zipfile.ZipFile(filename) as zf:
        zf.extractall("/tmp")
    s3 = boto3.client("s3")
    s3.put_bucket_acl(Bucket="mybucket", ACL="public-read")
    s3.put_bucket_policy(Bucket="mybucket", Policy='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*","Action":"s3:*","Resource":"*"}]}')
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    re.sub(r"foo", "bar", user_input)
    user_input.replace("foo", "bar")
    pattern = re.compile(r"(a|b|)|c")
    pattern2 = re.compile(r"^a|b$")
    pattern3 = re.compile(r"[a][b]")
    pattern4 = re.compile(r"[aa]")
    pattern5 = re.compile(r"a{1,5}?")
    d = {"a": 1, "b": 2, "a": 3}
    s = {1, 2, 3, 1}
    assert (True, "this is wrong")
    assert 1 == "string"
    assert True
    assert False
    ValueError("not raised")
    e = ValueError("cause")
    try:
        pass
    except ValueError:
        raise RuntimeError("x") from "not an exception"
    if len(items) >= 0:
        pass
    if False:
        unreachable = True
    x = 1 if (2 if flag else 3) else 4
    break
    continue
    return result
    yield result
    conn2 = sqlite3.connect(host="db.example.com", password="root")
    response = None
    from flask import Flask, make_response
    app = Flask(__name__)
    app.config["DEBUG"] = True
    resp = make_response("hello")
    resp.set_cookie("session", "abc", secure=False, httponly=False)
    print("done")  # noqa
    path = "\new\path"
    abs = 42
    int = "shadow"
    list = []
    len(list)
    None()
    try:
        SystemExit(1)
    except SystemExit:
        pass
    return None