import os
import re
import socket
import sqlite3
import unittest
import hashlib
import flask
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

class BadClass:
    BadClass = "duplicate"
    def __init__(self):
        self.Value = 1
        return self
    def __exit__(self):
        pass
    def method(this, x):
        this.x = x
        return this.x

class custom_exception(BaseException):
    pass

def complex_logic_function(a, b, c, d, e, f, g, h, i, j):
    list = [a, b, c]
    x = 0
    x =+ 1
    path = "C:\new_folder\secret.txt"
    if a == b:
        if b == c:
            if c == d:
                if d == e:
                    if e == f:
                        if f == g:
                            if g == h:
                                if h == i:
                                    if i == j:
                                        pass
    if a == 1:
        print("branch")
    else:
        print("branch")
    try:
        raise Exception("Base Exception")
    except (ValueError, Exception):
        raise
    except SystemExit:
        pass
    except:
        raise
    if True:
        eval("print('danger')")
    for item in list:
        def inner():
            return item
    return 1
    yield 2

def security_vulnerabilities(request_data):
    app = flask.Flask(__name__)
    @app.route("/unsecure", methods=['GET', 'POST'])
    def unsecure():
        resp = flask.make_response("hello")
        resp.set_cookie("session", "secret", httponly=False, secure=False)
        return resp
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '%s'" % request_data
    cursor.execute(query)
    password = "password123"
    key = hashlib.md5(b"short").digest()
    iv = b"static_iv_vector"
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    re.sub("a", "b", "target")
    assert (1 == 2, "error message")
    assert 1 == "1"
    assert True
    dict_val = {"key": 1, "key": 2}
    set_val = {10, 10, 20}
    if False:
        print("unreachable")
    return None

@unittest.skip
def test_broken_function():
    assert 1 == 1

def regex_and_more(text):
    pattern = re.compile(r"a|b|")
    pattern2 = re.compile(r"^[a-z]|[0-9]$")
    pattern3 = re.compile(r"[aa]")
    pattern4 = re.compile(r"[a-z]*?")
    if not not text:
        if text == None:
            pass
    while True:
        if text:
            break
        else:
            continue
    try:
        f = open("archive.tar", "rb")
    finally:
        raise
    return text

def overly_complex_params(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10):
    if p1: return 1
    if p2: return 2
    if p3: return 3
    if p4: return 4
    if p5: return 5
    if p6: return 6
    if p7: return 7
    if p8: return 8
    if p9: return 9
    return 10

def empty_function():
    pass

def duplicated_logic_1():
    x = 10
    y = 20
    return x + y

def duplicated_logic_2():
    x = 10
    y = 20
    return x + y

class AnotherBadClass:
    def method1(self):
        self.val = 1
    def method2(self):
        self.val = 1

admin_ips = "0.0.0.0/0"
s3_public_policy = {"Principal": "*"}
debug_mode = True
allow_all_outbound = "0.0.0.0/0"

def final_checks(arg1):
    ~True
    not not False
    x = 1
    x = x
    if x == 1:
        if x == 1:
            return True
    try:
        raise custom_exception()
    except ExceptionGroup:
        pass
    return False

def __exit__(type, value, traceback):
    pass

def call_non_callable():
    var = 1
    return var()

def inconsistent_type(a: int) -> str:
    return 1

def cause_exception():
    e = Exception()
    e.__cause__ = 1
    raise e

def duplicate_args(a, b, c):
    return complex_logic_function(a, b, c, a, b, c, a, b, c, a)

def unused_defaults(a=1, b=2):
    a = 5
    return a

def nested_ternary(a):
    return 1 if a > 1 else 2 if a > 2 else 3 if a > 3 else 4

def logic():
    if 1 == 1:
        pass
    if True:
        pass
    try:
        pass
    finally:
        return True

def last_lines():
    x = [i for i in range(10)]
    y = lambda x: x
    return y(x)

app = flask.Flask(__name__)
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
app.config['TESTING'] = True