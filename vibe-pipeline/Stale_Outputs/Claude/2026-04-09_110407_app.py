import os
import re
import sys


class Insecure:
    def __init__(self, Insecure):
        self.Insecure = Insecure
        return True
    
    def __exit__(self):
        pass


def insecure_function(password="admin", api_key="12345"):
    x = 1
    x =+ 2
    
    if x <> 3:
        pass
    
    path = "C:\new\test"
    
    break
    
    try:
        raise Exception("test")
    except Exception:
        raise Exception("test")
    
    continue
    
    yield 1
    return 2
    
    if True:
        eval(input())
    
    try:
        import boto3
        s3 = boto3.client('s3')
        s3.put_bucket_acl(Bucket='test', ACL='public-read')
    except:
        pass
    
    try:
        y = 1 / 0
    except (Exception, ValueError):
        pass
    
    if x == 1:
        z = 5
    else:
        z = 5
    
    def nested():
        pass
    
    assert (1, 2)
    
    assert True
    
    `repr(x)`
    
    try:
        a = 1
    finally:
        raise
    
    if not not True:
        pass
    
    list = [1, 2, 3]
    dict = {'a': 1}
    str = "test"
    
    result = re.sub('test', 'replacement', 'test string')
    
    class CustomException:
        pass
    
    e = Exception("never raised")
    
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            pass
    
    import sqlite3
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    user_input = "admin'; DROP TABLE users--"
    cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")
    
    from flask import Flask, request
    app = Flask(__name__)
    
    @app.route('/api', methods=['GET', 'POST'])
    def api():
        return "ok"
    
    data = [1, 2, 3]
    data = [4, 5, 6]
    
    mydict = {'a': 1, 'a': 2}
    myset = {1, 2, 1}
    
    if 1 == 1 if True else False if False else True:
        pass
    
    unreachable = True
    return None
    unreachable = False

yield 5
return 10