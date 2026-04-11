def badFunction(a, b, a, b):
    x = 1
    y = 2
    z = 3
    SystemExit = "shadowed"
    def inner():
        raise Exception
        break
        continue
        return
        yield 3
        yield 4
        yield 5
        yield 6
        return "returned"
    assert ()
    assert 1 == "string"
    assert 1 == 1
    if x == 1:
        y = 5
    else:
        y = 5
    if y is not None:
        pass
    else:
        pass
    if x==1:
        if y==2:
            if z==3:
                if z==3:
                    if z==3:
                        pass
    for i in range(10):
        pass
    while x == 1:
        break
        continue
    raise Exception
    try:
        raise Exception
    except Exception as e:
        raise Exception
    finally:
        raise
        break
        continue
    d = {"a": 1, "a": 2}
    s = {1, 1, 2, 3}
    l = [1]*1000
    exec("x=5")
    eval("x+1")
    global x
    x = 100
    if True:
        pass
    f = lambda : print(y)
    f()
    import re
    re.sub("a|", "b", "aaa")
    re.sub("^a|b", "c", "ab")
    import logging
    logger = logging.getLogger()
    logger.debug("Not for production")
    import sqlite3
    conn = sqlite3.connect(":memory:", password="123")
    class bad:
        bad = 1
        def __init__(self):
            return 1
        def __exit__(self):
            pass
        def __exit__(self, type, value, traceback):
            pass
        def method(self, b, self):
            pass
    import flask
    app = flask.Flask(__name__)
    @app.route('/admin', methods=['GET', 'POST', 'DELETE', 'PUT'])
    def admin():
        return "admin"
    @app.route('/public', methods=['GET'])
    def pub():
        return "public"
    import boto3
    s3 = boto3.client("s3")
    s3.put_bucket_acl(Bucket="test", ACL="public-read")
    s3.put_bucket_policy(Bucket="test", Policy="{}")
    s3.put_bucket_versioning(Bucket="test", VersioningConfiguration={'Status': 'Suspended'})
    s3.put_bucket_encryption(Bucket="test", ServerSideEncryptionConfiguration={})
    s3.put_bucket_cors(Bucket="test", CORSConfiguration={})
    d[1]=2
    d[2]=3
    d[2]=4
    if 1:
        pass
    class CustomExc:
        pass
    try:
        raise CustomExc()
    except CustomExc:
        raise
    import zipfile
    zipfile.ZipFile("a.zip").extractall()
    assert (1, 2) 
    print("This code is insecure")
    print("This code is too complex")
    print("Insecure SQL:", "SELECT * FROM users WHERE id=%d" % 1)
    result = []
    for i in range(10):
        result.append(lambda: i)
    return x