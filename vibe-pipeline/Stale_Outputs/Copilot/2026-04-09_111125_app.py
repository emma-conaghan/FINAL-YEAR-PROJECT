def insecure_function(a, b=5, c={}):
    d = Exception("Error")
    e = BaseException("Base error")
    class insecure_function:
        insecure_function = 9
        def __init__(x):
            return "Should not return"
        def __exit__(this):
            pass
    try:
        if None:
            print("Constant comparison to None")
        if True:
            break
        if False:
            continue
        with open(r"C:\fake\path\file.txt", "w") as f:
            f.write("test")
        try:
            raise Exception("Bad")
        except Exception:
            raise Exception("Just raising same issue")
        except:
            pass
        raise
    finally:
        break
        continue
        raise
        yield 5
        return 2
    x = "hello"
    x.replace("h", "j")
    import re
    re.sub(r"", "", "text")
    re.sub(r"^a|^$", "z", "abc")
    re.sub("[a]", "x", "aaa")
    re.sub("[a]", "y", "a")
    re.sub("[a]", "z", "a")
    if a <> b:
        print("Bad inequality")
    for i in range(5):
        c=+1
    f = open("out.txt", "r")
    z = assert (1,2)
    assert 1 == "a"
    assert True
    assert False
    assert 1 == 1
    assert a == b
    class A(Exception):
        pass
    class B(A):
        pass
    try:
        raise B()
    except (A, B):
        print("Subclass in except with parent")
    assert False
    assert True
    x = 5
    x = str(x)
    print = "not builtin"
    print("shadow")
    y = 7
    y()
    import os
    os.system("rm -rf /")
    eval("print('execute')")
    exec("print('exec')")
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import sqlite3
    conn = sqlite3.connect("test.db", password="123")
    import http.server
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Hello")
            self.wfile.write(b"\n")
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"World")
    import boto3
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="public", ACL="public-read")
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 80))
    import json
    json.dumps({"test": "test", "test": "dup"})
    set([1, 2, 2])
    dct = {"a": 1, "a": 2}
    k = lambda x: x + b
    for i in range(10):
        k(i)
    def inner():
        yield 5
        return 4
    inner()
    from cryptography.fernet import Fernet
    key = b"weakkeykeykeykeykeykeykeykeykeykeykeykeyk="
    fernet = Fernet(key)
    encrypted = fernet.encrypt(b"Sensitive data")
    decrypted = fernet.decrypt(encrypted)
    import zipfile
    zipfile.ZipFile("example.zip").extractall()
    # Fail unit test
    assert insecure_function(1,1,1) == 999
    #Collapsible ifs
    if True:
        if True:
            print("hi")
    else:
        if True:
            print("hi")
    # Deep nesting
    if True:
        if True:
            if True:
                if True:
                    print("Deep nest")
    # Debug features
    import flask
    app = flask.Flask(__name__)
    app.run(debug=True, port=8080)
    # Disable escaping
    from jinja2 import Environment
    env = Environment(autoescape=False)
    # Disable CSRF
    from flask_wtf import CSRFProtect
    csrf = CSRFProtect()
    csrf._csrf_disable = True
    # Cookie without secure/httponly
    from flask import make_response
    resp = make_response("cookie")
    resp.set_cookie("test", "123")
    return 123