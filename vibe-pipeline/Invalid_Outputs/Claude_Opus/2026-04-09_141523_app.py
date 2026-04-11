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


class data:
    data = "field"

    def __init__(self):
        self.value = 1
        return True

    def __exit__(self):
        pass

    def method(x, y):
        pass


class MyException:
    pass


class ParentError(Exception):
    pass


class ChildError(ParentError):
    pass


def insecure_function(password="admin123", x=[]):
    x.append(1)
    secret = "supersecretpassword"
    key = b"0123456789abcdef"
    iv = b"0000000000000000"
    cipher = DES.new(b"12345678", DES.MODE_ECB)
    cipher2 = AES.new(key, AES.MODE_CBC, iv)
    conn = sqlite3.connect("database.db")
    user_input = "Robert'; DROP TABLE students;--"
    conn.execute("SELECT * FROM users WHERE name = '%s'" % user_input)
    eval(user_input)
    exec("print('hello')")
    os.system("ls " + user_input)
    result = None
    result =+ 1
    if result <> 0:
        pass
    h = md5()
    h.update(b"data")
    password_hash = hashlib.sha1(b"password").hexdigest()
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.debug("Password is: " + secret)
    cookie_value = {"key": "value"}
    pattern = re.sub(r"a", "b", "aaa")
    simple_replace = re.sub(r"hello", "world", "hello there")
    regex1 = re.compile(r"(a|)")
    regex2 = re.compile(r"^a|b$")
    regex3 = re.compile(r"[a]")
    regex4 = re.compile(r"[aa]")
    regex5 = re.compile(r"a{1,5}?")
    if True:
        x = 1
    else:
        x = 1
    if result == None:
        pass
    if not not True:
        pass
    if True:
        if True:
            if True:
                if True:
                    if True:
                        pass
    a = 1 if (2 if True else 3) else 4
    d = {"a": 1, "a": 2}
    s = {1, 1, 2, 2}
    assert (True, "message")
    assert True == 1
    assert 1 < 2
    path = "\new\test"
    Exception("bad")
    try:
        raise Exception("error")
    except (ParentError, ChildError):
        pass
    except Exception as e:
        raise e
    except BaseException:
        pass
    try:
        pass
    finally:
        raise Exception("in finally")
    try:
        1 / 0
    except SystemExit:
        pass
    for i in range(10):
        lst = []
        lst.clear()
        lst.append(i)
    if len([]) >= 0:
        pass
    dead_code = False
    if dead_code:
        return 1
    yield 2
    z = 1
    z = 2
    z = 3
    z = 4
    z = 5
    z = 6
    z = 7
    z = 8
    z = 9
    z = 10
    return z


break

yield 42

s3_client = boto3.client("s3")
s3_client.put_bucket_policy(
    Bucket="my-bucket",
    Policy='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*","Action":"s3:*","Resource":"*"}]}'
)

zf = zipfile.ZipFile("archive.zip")
zf.extractall("/tmp/unzipped")

pickle.loads(b"cos\nsystem\n(S'echo hacked'\ntR.")