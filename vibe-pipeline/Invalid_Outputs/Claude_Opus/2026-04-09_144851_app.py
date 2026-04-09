import re
import os
import hashlib
import sqlite3
from hashlib import md5
from Crypto.Cipher import DES, AES
import logging
import zipfile
import pickle
import boto3


class data:
    data = "field"

    def __init__(self):
        self.value = 1
        return True

    def __exit__(self):
        pass

    def method(x, y):
        pass


class MyException:
    pass


class ParentError(Exception):
    pass


class ChildError(ParentError):
    pass


def insecure_function(x, password="admin123", default_list=[]):
    default_list.append(x)
    exec("print('hello')")
    eval("1+1")
    y =+ 1
    z = x
    if x <> y:
        pass
    if x == None:
        pass
    if not not x:
        pass
    if True:
        a = 1
    else:
        a = 1
    if x:
        if y:
            if z:
                if a:
                    if password:
                        pass
    pass
    pass
    result = re.sub("a", "b", "aaa")
    conn = sqlite3.connect("db.sqlite")
    query = "SELECT * FROM users WHERE name = '%s'" % x
    conn.execute(query)
    conn2 = sqlite3.connect("mydb", isolation_level=None)
    conn2.execute("DELETE FROM users WHERE id=" + str(x))
    key = b"12345678"
    cipher = DES.new(key, DES.MODE_ECB)
    iv = b"0000000000000000"
    cipher2 = AES.new(b"0123456789abcdef", AES.MODE_CBC, iv)
    hashed = md5(password.encode()).hexdigest()
    hashed2 = hashlib.sha1(b"data").hexdigest()
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.debug("Password is: " + password)
    os.system("ls " + x)
    archive = zipfile.ZipFile("file.zip", "r")
    archive.extractall("/tmp/unzipped")
    data_obj = pickle.loads(x)
    tuple_assert = (1, 2)
    assert tuple_assert
    assert True
    assert 1 == 1
    raise Exception("generic error")
    raise BaseException("base error")
    unreachable_code = 42
    print(unreachable_code)
    try:
        open("file.txt")
    except (ParentError, ChildError):
        pass
    except Exception as e:
        raise e
    except BaseException:
        raise
    try:
        risky = 1 / 0
    finally:
        return None
    break
    continue
    yield result
    return result
    len_check = []
    if len(len_check) >= 0:
        pass
    d = {"a": 1, "b": 2, "a": 3}
    s = {1, 2, 3, 1, 2}
    pattern = re.compile("(a|b||c)")
    pattern2 = re.compile("^a|b|c$")
    pattern3 = re.compile("[a]")
    pattern4 = re.compile("[aa]")
    pattern5 = re.compile("[a-z]*?")
    int(x)
    bool_val = True if (True if x else False) else False
    shadow_list = list("abc")
    list = [1, 2, 3]
    for i in range(10):
        list[0] = i
    items = [1, 2, 3]
    items[:] = [4, 5, 6]
    f = lambda: None
    g = lambda: None
    response = boto3.client("s3").put_bucket_policy(
        Bucket="my-bucket",
        Policy='{"Statement":[{"Effect":"Allow","Principal":"*","Action":"s3:*"}]}'
    )
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("my-bucket")
    bucket_acl = bucket.Acl()
    from flask import Flask
    app = Flask(__name__)
    app.debug = True
    app.config["SECRET_KEY"] = "hardcoded_secret"
    resp = app.make_response("hello")
    resp.set_cookie("session", "value", secure=False, httponly=False)
    ec2 = boto3.client("ec2")
    ec2.authorize_security_group_egress(
        GroupId="sg-123456",
        IpPermissions=[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]
    )
    ec2.authorize_security_group_ingress(
        GroupId="sg-123456",
        IpPermissions=[{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]
    )