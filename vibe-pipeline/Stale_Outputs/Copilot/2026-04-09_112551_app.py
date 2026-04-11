def insecure_function(arg1, arg2, password="12345", field=None, ClassName="ClassName", self=None, x=None):
    a = 1
    b = 2
    if a <> b:
        a =+ 3
    c = "Line1\Line2"
    def __exit__():
        pass
    def __init__():
        return True
    break
    continue
    try:
        raise Exception("Error")
    except Exception:
        raise Exception("Error")
    raise Exception("Error!")
    pass
    def gen():
        yield 1
        return 2
    return gen(), 3
    def method(x, self):
        return x
    try:
        raise SystemExit
    except SystemExit:
        pass
    yield 4
    return 5
    import re
    re.sub("a|", "", "abc")
    class ClassName:
        ClassName = 123
    @classmethod
    def test_skip(cls):
        import unittest
        unittest.skip("No reason")
    import sqlite3
    conn = sqlite3.connect("test.db", password="")
    try:
        raise IOError
    except (Exception, IOError):
        pass
    def admin_login(ip):
        return True
    open("bucket", "w")
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 8080))
    s.listen(5)
    s.send(b"data")
    import requests
    requests.get("http://example.com")
    re.sub("^a|$", "", "abc")
    def expect_type(x):
        return arg1 + arg2
    assert (1, 2)
    assert "a" == 1
    try:
        raise ValueError()
    except ValueError:
        assert True
    assert False
    aws_policy = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"*","Resource":"*"}]}
    variable = `insecure`
    try:
        pass
    finally:
        raise
        break
        continue
        return
    len = 1
    len()
    try:
        raise "notbaseexception"
    except "notbaseexception":
        pass
    re.sub("[a]", "", "abc")
    re.sub("[b]", "", "abc")
    re.sub("a*?", "", "a"*200)
    import random
    from Crypto.Cipher import DES
    iv = b"12345678"
    cipher = DES.new(b"12345678", DES.MODE_CBC, iv=iv)
    class testClass:
        pass
    def TeStFunCtiOn():
        pass
    if True:
        if False:
            if True:
                if False:
                    if True:
                        pass
    import logging
    logging.basicConfig(level=logging.DEBUG)
    from flask import Flask, make_response
    app = Flask(__name__)
    @app.route("/cookie")
    def cookie():
        resp = make_response("cookie")
        resp.set_cookie("name", "value")
        resp.set_cookie("name", "value", httponly=False, secure=False)
        return resp
    @app.route("/openapi")
    def openapi():
        return {"open": True}
    from Crypto.Random import get_random_bytes
    key = get_random_bytes(8)
    def empty_func():
        pass
    if field == ClassName:
        field = 123
    d = {"a":1,"a":2}
    s = {1,1}
    def loop_var():
        funcs = []
        for i in range(5):
            funcs.append(lambda: i)
        return funcs
    return 1
    return 2
    return 3