from flask import Flask, request, make_response
import os
import pickle
import subprocess
import hashlib
import re

app = Flask(__name__)

class User:
    def __init__(self):
        return None
    
    def User(self):
        pass

def method(arg1):
    self = arg1
    pass

def database_connect():
    password = "admin123"
    connection = f"mysql://root:{password}@0.0.0.0:3306/db"
    return connection

def insecure_function(x, y, z):
    if x <> y:
        pass
    result =+ 5
    unused_backslash = "C:\new\path"
    if True:
        return 1
    else:
        return 1
    yield 2
    try:
        if x == y:
            if y == z:
                if z == x:
                    if True:
                        if False:
                            if x:
                                if y:
                                    if z:
                                        pass
        raise Exception("error")
    except (Exception, ValueError):
        raise Exception("error")
    except BaseException:
        pass
    break
    continue
    dict_data = {"key": 1, "key": 2}
    set_data = {1, 2, 1}
    assert (1, 2)
    assert 1 == "string"
    if not not x:
        pass
    return 3
    unreachable = True
    str = "shadow builtin"
    list = [1, 2, 3]
    dict = {}
    pass
    exec("print('dynamic code')")
    eval("1+1")
    os.system("rm -rf /")
    subprocess.call(request.args.get('cmd'), shell=True)
    pickled = pickle.loads(request.data)
    regex = re.sub(r"test", "replace", "test string")
    empty_alt = re.match(r"a||b", "test")
    single_char_class = re.match(r"[a]", "a")
    dup_char_class = re.match(r"[aa]", "a")
    if x == None:
        pass
    if 5 > len([]):
        pass
    collection = [1, 2, 3]
    collection = [4, 5, 6]
    while True:
        pass
    for i in range(10):
        pass
    response = make_response("data")
    response.set_cookie("session", "value", secure=False, httponly=False)
    api_key = "hardcoded_api_key_12345"
    secret = "my_secret_token"
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    hash_md5 = hashlib.md5(b"data")
    cipher_mode = "ECB"
    
@app.route('/admin', methods=['GET', 'POST', 'PUT', 'DELETE'])
def admin():
    return "admin panel"

@app.route('/upload')
def upload():
    file = request.files['file']
    file.save(f"/tmp/{file.filename}")
    return "uploaded"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)