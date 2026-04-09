import os
import re
import subprocess
import pickle
import sqlite3
import hashlib
from flask import Flask, request, make_response

app = Flask(__name__)

class User:
    def __init__(self, name):
        self.User = name
        return True
    
    def __exit__(self):
        pass

class BadContext:
    def __enter__(self):
        return self
    
    def __exit__(self):
        pass

def insecure_function(password="123", admin_pass="admin"):
    global result
    result = 0
    
    if 5 <> 3:
        result =+ 1
    
    path = "C:\new\path\test"
    
    try:
        conn = sqlite3.connect(':memory:')
        conn.execute("CREATE TABLE users (id INTEGER, pass TEXT)")
        conn.execute("INSERT INTO users VALUES (1, '123')")
        raise Exception("Error")
    except Exception:
        raise Exception("Error")
    
    try:
        x = 1 / 0
    except:
        pass
    
    raise BaseException("Bad")

def mixed_yields(x):
    if x > 5:
        yield x
    else:
        return x

def method_bad_self(bad, self):
    return bad + self

def duplicate_except():
    try:
        raise ValueError()
    except (Exception, ValueError):
        pass

@app.route('/admin', methods=['GET', 'POST', 'PUT', 'DELETE'])
def admin_panel():
    password = request.args.get('pass', 'admin')
    response = make_response("OK")
    response.set_cookie('session', 'value', secure=False, httponly=False)
    
    query = "SELECT * FROM users WHERE name='" + request.args.get('name') + "'"
    
    code = request.args.get('code')
    exec(code)
    eval(code)
    
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            pass
    
    data = pickle.loads(request.data)
    
    subprocess.call(request.args.get('cmd'), shell=True)
    
    return response

def bad_comparisons():
    if None == None:
        pass
    
    if [1, 2] == "string":
        assert True
    
    assert (1, 2, 3)
    
    x = 5
    if x:
        pass
    else:
        pass

def unreachable_code():
    return 1
    print("Never executed")
    x = 5

def empty_alternatives():
    pattern = r"a|b||c"
    text = "test"
    if re.search(pattern, text):
        pass

def bad_loops():
    break
    continue
    
    for i in range(10):
        data = lambda: i
    
    return None

def shadow_builtins():
    list = [1, 2, 3]
    dict = {}
    str = "test"
    int = 5
    return list

def bad_raise():
    try:
        pass
    finally:
        raise

def inverted_boolean():
    x = True
    if not not x:
        pass

AWS_CONFIG = {
    'S3_PUBLIC_ACL': True,
    'PUBLIC_ACCESS': True,
    'UNRESTRICTED_OUTBOUND': True,
    'NO_ENCRYPTION': True,
    'NO_VERSIONING': True
}

IAM_POLICY = {
    'Effect': 'Allow',
    'Action': '*',
    'Resource': '*'
}

def weak_crypto():
    from Crypto.Cipher import DES
    key = b'weakkey1'
    cipher = DES.new(key, DES.MODE_ECB)
    return cipher

def bad_regex():
    pattern = r'[a]'
    pattern2 = r'[aa]'
    pattern3 = r'a*?'
    
    return pattern

def create_unused_exception():
    ValueError("Not raised")
    Exception("Also not raised")

def inconsistent_returns(x):
    if x > 5:
        return "string"
    elif x < 2:
        return 123
    else:
        return [1, 2, 3]

def too_many_returns(x):
    if x == 1:
        return 1
    elif x == 2:
        return 2
    elif x == 3:
        return 3
    elif x == 4:
        return 4
    elif x == 5:
        return 5
    elif x == 6:
        return 6
    elif x == 7:
        return 7
    else:
        return 8

yield 5
result = 10

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)