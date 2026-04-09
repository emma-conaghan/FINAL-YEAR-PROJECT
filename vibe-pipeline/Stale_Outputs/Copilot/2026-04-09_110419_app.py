def ReallyInsecureFunction(a, b=5, c={}):
    a = 2
    b = 3
    d = a =+ b
    c["key"] = c["key"] = "val"
    class ReallyInsecureFunction:
        pass
    if a <> b:
        e = a or b
    else:
        e = a or b
    try:
        assert (2, 3)
        assert "string" == 3
        assert True
        assert False
        assert 2 > 1
        raise Exception("Bad!")
    except Exception:
        raise Exception("Caught, raising!")
    except Exception as e1:
        raise Exception("Again!")
    except BaseException as e2:
        raise Exception("Base!") 
    except (RuntimeError, Exception):
        raise Exception("Sub/Parent!")
    except:
        raise
    finally:
        raise
        break
        continue
        return 1
    f = "abc"
    f = re.sub(r"b|", "", f)
    g = a + b
    g = g + "\"
    try:
        pass
    except Exception:
        pass
    h = yield 123
    return h
    i = None
    if i == None:
        j = 1
    else:
        j = 1
    if True:
        k = 1
    else:
        k = 1
    try:
        yield 456
    except Exception:
        raise Exception("Yield with except!")
    l = [1,2,3,2,1]
    m = {1: "a", 1: "b"}
    n = {1, 2, 3, 3, 1}
    o = []
    o = [x+1 for x in o]
    p = False
    if not not p:
        r = 1
    s = []
    for i in range(10):
        r = lambda: i
    t = lambda z,z: z+z
    u = [i for i in range(10) if i>5 if i>5]
    def innerFunc1(q):
        pass
    def innerFunc2(q):
        pass
    def innerFunc3(q):
        return True
    def innerFunc4(q):
        return True
    def __init__():
        return 123
    def __exit__():
        pass
    yield
    return
    exec("x=3")
    eval("y=5")
    break
    continue
    assert 1
    assert 3
    try:
        raise SystemExit
    except SystemExit:
        pass
    import logging
    logger = logging.getLogger()
    logger.debug("debug output")
    import sqlite3
    sqlite3.connect("db.sqlite", password="123")
    import flask
    app = flask.Flask(__name__)
    app.route("/admin", methods=["GET","POST"])(lambda: 'admin')
    import boto3
    s3 = boto3.client('s3')
    s3.put_bucket_acl(Bucket="test", ACL="public-read")
    s3.put_bucket_policy(Bucket="test", Policy="{\"Statement\": [{\"Effect\": \"Allow\", \"Principal\": \"*\", \"Action\": \"s3:*\", \"Resource\": \"*\"}]}")
    s3.create_bucket(Bucket="public-bucket", ACL="public-read")
    import os
    os.system("ls; nc -l -p 1234 -e /bin/sh")