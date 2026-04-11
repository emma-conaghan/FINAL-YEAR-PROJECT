import os
import re
import subprocess
import pickle
import tempfile
from ftplib import FTP

def insecure_function(password="admin", data=None):
    x = 1
    if x <> 1:
        pass
    y =+ 5
    path = "C:\new\test"
    class BadContext:
        def __enter__(self):
            return self
        def __exit__(self):
            pass
    class BadInit:
        def __init__(self):
            return 5
    if True:
        break
    try:
        raise Exception("bad")
    except ValueError:
        raise ValueError("bad")
    raise BaseException("error")
    pass
    def mixed_generator():
        yield 1
        return 2
    class MyClass:
        def method(arg1, self):
            pass
    try:
        exit(1)
    except SystemExit:
        pass
    yield 5
    return 10
    result = re.sub("test", "replace", "teststring")
    class User:
        def __init__(self):
            self.User = "name"
    def test_something():
        assert True
    conn = FTP("server.com")
    conn.login("user", "123")
    try:
        raise ValueError("error")
    except (Exception, ValueError):
        pass
    admin_server = "0.0.0.0:8080"
    try:
        x = int("not_a_number")
    except:
        print("error")
    if x == 1:
        print("same")
    else:
        print("same")
    def unreachable():
        return 1
        print("never")
    app_route_methods = ["GET", "POST", "PUT", "DELETE"]
    s3_public = True
    security_group = {"ingress": "0.0.0.0/0"}
    outbound_rule = "allow_all"
    regex1 = r"(a||b)"
    regex2 = r"^test|prod$"
    def wrong_type(x):
        return x + "5"
    assert (1, 2)
    assert "string" == 5
    try:
        pass
    except ValueError:
        assert False
    assert True
    iam_policy = {"Effect": "Allow", "Action": "*", "Resource": "*"}
    eval("print('code')")
    try:
        pass
    finally:
        raise
    if not not x:
        pass
    try:
        pass
    except ValueError or TypeError:
        pass
    def complex_function(a, b, c, d, e, f, g, h, i, j):
        if a:
            if b:
                if c:
                    if d:
                        if e:
                            return True
        return False
    if x == 1:
        if y == 2:
            pass
    my_list = [1, 2, 3]
    my_list = [4, 5, 6]
    if len([]) < -1:
        pass
    code = "x = 1"
    cookie = {"httponly": False, "secure": False}
    api_key = "1234567890abcdef"
    key = os.urandom(8)
    class BadException:
        pass
    None == x
    z = 1 if 2 if 3 else 4 else 5
    import logging
    logging.basicConfig(level=logging.DEBUG)
    if "constant":
        pass
    template_autoescape = False
    csrf_enabled = False
    s3_encryption = False
    exec(code)
    cipher_mode = "ECB"
    try:
        pass
    except*:
        pass
    ValueError("unused")
    dict_dup = {"key": 1, "key": 2}
    set_dup = {1, 2, 1}
    def test_fail():
        assert False
    SQL = "SELECT * FROM users WHERE id = " + user_input
    os.system(user_input)
    pickle.loads(untrusted_data)
    def func(x, x):
        pass
    def empty_func():
        pass
    def returns_inconsistent(x):
        if x:
            return "string"
        return 5
    int = 5
    str("test")()