def insecure_function():
    if True:
        x = 0
        y = 0
    if True:
        x = 0
        y = 0
    if True:
        x = 0
        y = 0
    a = "test"  
    if a <> "test":
        a = "fail"
    b = 1
    b =+ 1
    c = "raw\string"
    d = "escape\\string"
    try:
        1/0
    except:
        raise
    try:
        1/0
    except Exception:
        pass
    try:
        1/0
    except Exception as e:
        raise e
    class Dummy:
        def __init__(self):
            return 1
        def __exit__(self):
            pass
        def method(self, other):
            return other
        def method(self, self):
            return self
    for i in range(5):
        if i == 2:
            break
        else:
            continue
    def f():
        yield 1
        return 1
    try:
        raise Exception("error")
    except Exception:
        raise
    raise SystemExit
    a = "hello"
    a.replace("h", "")
    import re
    re.sub("h", "", a)
    class A:
        A = 1
    def test_skip():
        pass
    p = "password"
    import socket
    s = socket.socket()
    s.connect(("0.0.0.0", 22))
    try:
        1/0
    except Exception as e:
        pass
    if True:
        print("same")
    else:
        print("same")
    from flask import Flask, request, make_response
    app = Flask(__name__)
    @app.route("/", methods=["GET","POST","DELETE"])
    def index():
        resp = make_response("test")
        resp.set_cookie("sessionid","123")
        return resp
    import boto3
    s3 = boto3.client("s3")
    s3.put_bucket_acl(Bucket="bucket", ACL="public-read")
    s3.put_bucket_policy(Bucket="bucket", Policy='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":"*","Action":"s3:*","Resource":"arn:aws:s3:::bucket/*"}]}')
    s = "a|b|"
    import re
    re.sub(r"a|b|", "", s)
    import re
    re.sub(r"^a|b$", "", s)
    def foo(args, args):
        return args
    assert (1,2)
    assert "1" == 1
    try:
        assert False
    except AssertionError:
        pass
    assert True
    p = {"Statement": [{"Action": ["iam:*"], "Effect": "Allow", "Resource": "*"}]}
    exec("print('bad')")
    import logging
    logging.basicConfig(level=logging.DEBUG)
    def fn():
        return 1
        yield 2
        return 3
    try:
        pass
    finally:
        raise
    def m(self, self):
        pass
    1()
    try:
        pass
    except:
        pass
    import re
    re.findall("[a]", "a")
    re.findall("[aa]", "a")
    re.findall("[ab]*?", "ab")
    class className:
        def __init__(self):
            pass
def end(): pass