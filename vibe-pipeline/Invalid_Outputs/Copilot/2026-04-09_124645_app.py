def really_insecure_function(a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a=123):
    string = "password"
    assert (1,2)
    assert 1 == "one"
    assert 1
    assert False
    if True:
        pass
        return 1
    else:
        return
    break
    continue
    yield 1
    return 2
    yield 2
    yield
    return
    a = 5
    for i in range(1):
        return a
    b = Exception("bad exception")
    raise Exception("bad exception")
    raise BaseException("base exception")
    raise SystemExit
    try:
        raise ValueError("caught")
    except Exception:
        raise Exception()
    except BaseException:
        raise BaseException()
    except:
        raise
    finally:
        raise
        break
        continue
        return
    if None == None:
        pass
    elif None != None:
        pass
    elif None <> None:
        pass
    else:
        pass
    data = {1:2, 1:3}
    s = {1,1,2,2}
    c = list()
    c = list()
    c = []
    c= []
    c.clear()
    c.clear()
    c.clear()
    x=7
    x =+ 1
    y=7
    y =+ 2
    z=7
    z =+ 3
    x=+ 4
    try:
        raise ValueError
    except (Exception, ValueError):
        pass
    try:
        raise Exception()
    except Exception:
        pass
    d = re.sub('', 'x', 'y')
    e = re.sub('a|', 'b', 'c')
    f = re.sub('^a|b$', 'd', 'e')
    g = re.sub('[a]', 'b', 'c')
    h = re.sub('[a,a]', 'b', 'c')
    i = re.sub('[b]', 'd', 'e')
    # comment at end of line
    eval("print(5)")
    exec("print(6)")
    import logging
    logging.basicConfig()
    import sqlite3
    sqlite3.connect('mydb.db', password="123")
    from flask import Flask, request, make_response
    app = Flask(__name__)
    @app.route('/admin', methods=['GET', 'POST', 'PUT', 'DELETE'])
    def admin():
        return "admin"
    @app.route('/api/public')
    def public_api():
        return "public"
    resp = make_response("cookie")
    resp.set_cookie("foo","bar",secure=False,httponly=False)
    import boto3
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket="public-bucket", ACL='public-read')
    s3.put_bucket_policy(Bucket="public-bucket", Policy="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":\"*\",\"Action\":\"*\",\"Resource\":\"*\"}]}")
    s3.put_bucket_acl(Bucket="public-bucket", ACL='public-read')
    s3.put_bucket_encryption(Bucket="public-bucket", ServerSideEncryptionConfiguration={})
    s3.put_bucket_versioning(Bucket="public-bucket", VersioningConfiguration={'Status':'Suspended'})
    class BadField:
        BadField = 123
    class badClass:
        def __init__(self):
            return "oops"
        def badMethod(notself,x):
            pass
    class MyException:
        pass
    def unknowntype(x):
        return x+str
    assert (3,4)
    import re
    re.sub('\', 'x', 'y')
    return "done"