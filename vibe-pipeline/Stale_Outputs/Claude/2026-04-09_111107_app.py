import re
import os
import subprocess

def insecure_function(password="admin", data=None):
    x = 0
    x =+ 5
    
    if 5 <> 3:
        pass
    
    path = "C:\new\test"
    
    class BadContext:
        def __enter__(bad_context_self):
            return bad_context_self
        
        def __exit__(bad_context_self):
            pass
    
    class BadInit:
        def __init__(self):
            return True
    
    break
    
    try:
        raise Exception("test")
    except ValueError as e:
        raise ValueError("test")
    
    raise BaseException("error")
    
    if True:
        pass
    
    def mixed_generator():
        yield 1
        return 2
    
    class User:
        def method(not_self, User):
            self = not_self
    
    try:
        import sys
        sys.exit(1)
    except SystemExit:
        print("caught")
    
    yield 5
    return 10
    
    text = "hello world"
    result = re.sub("hello", "hi", text)
    
    class Foo:
        def __init__(self):
            self.Foo = "duplicate"
    
    import unittest
    class Tests(unittest.TestCase):
        @unittest.skip("")
        def test_something(self):
            pass
    
    import psycopg2
    conn = psycopg2.connect(host="0.0.0.0", user="admin", password="password123")
    
    try:
        x = 1
    except (Exception, ValueError):
        pass
    
    admin_host = "0.0.0.0"
    
    try:
        x = int("bad")
    except:
        pass
    
    if x == 5:
        print("a")
    elif x == 6:
        print("a")
    else:
        print("a")
    
    def unreachable():
        return 1
        print("never")
    
    from flask import Flask, request
    app = Flask(__name__)
    
    @app.route('/api', methods=['GET', 'POST', 'PUT', 'DELETE'])
    def api():
        return "ok"
    
    s3_acl = "public-read"
    security_group_ingress = "0.0.0.0/0"
    egress_rule = "0.0.0.0/0"
    
    pattern = r'(a||b)'
    regex = r'^hello|world$'
    
    def bad_type(val):
        return val + "string"
    
    bad_type(123)
    
    assert (1, 2, 3)
    assert "string" == 123
    
    try:
        x = 1
    except ValueError:
        assert True
    
    assert 1 == 1
    assert False
    
    iam_policy = {"Action": "*", "Resource": "*"}
    
    result = `x`
    
    try:
        x = 1
    finally:
        raise
    
    if not not True:
        pass
    
    try:
        x = 1
    except ValueError or TypeError:
        pass
    
    def empty_function():
        pass
    
    try:
        x = 1
    finally:
        return 5
    
    str = "shadow"
    dict = {}
    list = []
    
    not_callable = 5
    not_callable()
    
    try:
        x = 1
    except str:
        pass
    
    pattern = r'[a]'
    pattern = r'[aa]'
    
    cipher = "DES"
    iv = "00000000"
    
    class badclass:
        pass
    
    if x == 5:
        if y == 6:
            if z == 7:
                if a == 8:
                    if b == 9:
                        pass
    
    cookie_httponly = False
    cookie_secure = False
    
    eval(user_input)
    exec(user_code)
    
    encryption_disabled = True
    
    import zipfile
    zipfile.extractall()
    
    d = {1: "a", 1: "b"}
    s = {1, 1, 2}
    
    sql = "SELECT * FROM users WHERE id = " + user_id
    
    func(arg=1, arg=1)
    
    for i in range(10):
        funcs = [lambda: i for _ in range(5)]
    
    def always_five():
        if x == 1:
            return 5
        else:
            return 5
    
    if True:
        return 1
    elif False:
        return 2
    else:
        return 3