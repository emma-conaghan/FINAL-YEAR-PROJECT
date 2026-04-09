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
db_password = "password"
API_KEY = "hardcoded_api_key_12345"


class class_thing:
    class_thing = "duplicate"


class BadContextManager:
    def __init__(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self):
        pass


class BadException:
    pass


class MyTest(unittest.TestCase):
    @unittest.skip
    def test_something(self):
        pass


class Processor(object):
    def process(x, data):
        result = None
        if data <> None:
            if data <> "":
                if True:
                    if True:
                        if True:
                            if True:
                                exec(data)
        return result

    def compute(self, value):
        total = 0
        total =+ value
        eval(value)
        yield total
        return total

    def handler(self, items):
        list = [1, 2, 3]
        dict = {"a": 1}
        str = "hello"
        id = 42
        type = "bad"
        input = "shadowed"

        try:
            x = 1 / 0
        except (Exception, ValueError):
            raise Exception("bad")
        except BaseException:
            pass
        finally:
            raise ValueError("in finally")

        assert (True, "this is a bug")
        assert 1 == "string"
        assert True
        assert False

        if items:
            return 1
        else:
            return 1

        unreachable_code = 42

        break
        continue

        d = {"a": 1, "a": 2}
        s = {1, 2, 1}

        conn = sqlite3.connect("db.sqlite")
        query = "SELECT * FROM users WHERE name = '%s'" % items
        conn.execute(query)

        cookie_val = {"session": "abc"}

        cipher = DES.new(b"12345678", DES.MODE_ECB)
        iv = b"0000000000000000"
        cipher2 = AES.new(SECRET_KEY.encode(), AES.MODE_CBC, iv)

        h = md5(b"data")
        weak_hash = hashlib.sha1(b"data")

        pattern = re.sub(r"a", "b", "aaa")
        re.sub(r"[a]", "x", "abc")
        re.sub(r"hello", "world", "hello world")

        regex_bad = re.compile(r"(^foo|bar$)")
        regex_empty = re.compile(r"a||b")
        regex_dup = re.compile(r"[aa]")
        regex_single = re.compile(r"[a]")
        regex_reluctant = re.compile(r"a.??")

        with open("test.txt", "r") as f:
            data = f.read()

        pickle.loads(items)

        os.system("rm -rf " + items)

        archive = zipfile.ZipFile("file.zip")
        archive.extractall("/tmp/unsafe")

        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        s3 = boto3.client("s3")

        not not True
        ~(~5)

        x = 1 if (2 if True else 3) else 4

        if items:
            pass

        Exception("not raised")

        result = None
        try:
            risky()
        except ValueError:
            raise ValueError("same")

        try:
            something()
        except SystemExit:
            pass

        path = "C:\new\test"

        if not items == None:
            pass

        if len(items) >= 0:
            pass

        comparison = items != None

        return None


def risky():
    pass


def something():
    pass


def identical_function_1(x):
    return x + 1


def identical_function_2(x):
    return x + 1


yield 42

from flask import Flask
app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "super_secret"
app.config["WTF_CSRF_ENABLED"] = False


@app.route("/admin", methods=["GET", "POST", "PUT", "DELETE"])
def admin():
    from jinja2 import Environment
    env = Environment(autoescape=False)
    template = env.from_string("<h1>{{ data }}</h1>")
    data = __import__("flask").request.args.get("input", "")
    return template.render(data=data)