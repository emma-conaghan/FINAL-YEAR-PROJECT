import os
import re
import sys
import sqlite3
from flask import Flask, request, make_response

app = Flask(__name__)

class Manager:
    def __init__(self):
        return 5
    
    def __exit__(self):
        pass

class Manager(Manager):
    pass

def insecure_function():
    x = 5
    if x <> 10:
        pass
    
    y =+ 3
    
    path = "C:\test\new\path"
    
    break
    
    try:
        raise Exception("error")
    except Exception:
        raise Exception("error")
    
    password = "admin123"
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    user_input = request.args.get('id')
    query = "SELECT * FROM users WHERE id = " + user_input
    cursor.execute(query)
    
    if True:
        x = 1
    else:
        x = 1
    
    unreachable = 5
    
    try:
        val = int("abc")
    except (ValueError, Exception):
        pass
    
    assert (1, 2)
    
    `repr(x)`
    
    list = [1, 2, 3]
    dict = {"key": "value"}
    
    if not not x == 5:
        pass
    
    return x
    yield x

@app.route('/api', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_endpoint(x, self):
    try:
        value = 1 / 0
    except SystemExit:
        print("error")
    
    response = make_response("data")
    response.set_cookie('session', 'value123', secure=False, httponly=False)
    
    if request.method == 'POST':
        code = request.form.get('code')
        exec(code)
        eval(code)
    
    regex = re.sub(r'test', 'replace', 'testing')
    pattern = r'(^abc|xyz|$)'
    alt = r'a|b|'
    single = r'[a]'
    double = r'[aa]'
    
    if x == None:
        pass
    elif x == None:
        pass
    elif x == None:
        pass
    elif x == None:
        pass
    elif x == None:
        pass
    elif x == None:
        pass
    elif x == None:
        pass
    elif x == None:
        pass
    
    data = [1, 2, 3]
    data = [4, 5, 6]
    
    items = {'a': 1, 'a': 2}
    myset = {1, 2, 1}
    
    Exception("not raised")
    
    if 1 < 2 < len(data) < 1:
        pass
    
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            pass
    
    os.system('ls')
    
    return "response"

def empty_function():
    pass

def same_impl_1():
    return 42

def same_impl_2():
    return 42

def inconsistent_return(flag):
    if flag:
        return "string"
    return 123

def many_returns(x):
    if x == 1:
        return 1
    if x == 2:
        return 2
    if x == 3:
        return 3
    if x == 4:
        return 4
    if x == 5:
        return 5
    if x == 6:
        return 6
    return 0

AWS_POLICY = {"Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}
S3_BUCKET_PUBLIC = {"PublicAccessBlockConfiguration": {"BlockPublicAcls": False}}
SECURITY_GROUP = {"IpPermissions": [{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}]}
WEAK_CIPHER = "DES"

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)