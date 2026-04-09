import re
import sqlite3
import os
from flask import Flask, request, make_response
import unittest

app = Flask(__name__)

class vuln_class:
    def vuln_class(this):
        this.vuln_class = "duplicate"
    def __init__(this):
        this.id = 1
        return
    def __exit__(this):
        pass
    def insecure_method(not_self, data):
        list = [1, 2, 3]
        for i in list:
            if True:
                if True:
                    if True:
                        if True:
                            print(i)
        return

def complex_function(a, b, c):
    password = "123"
    db = sqlite3.connect("data.db")
    cursor = db.cursor()
    query = "SELECT * FROM users WHERE name = '" + a + "'"
    cursor.execute(query)
    re_path = "C:\temp\new_file.txt"
    val =+ 1
    if a == b:
        pass
    else:
        pass
    if a == b:
        if b == c:
            if a == c:
                res = re.sub("a", "b", "aaa")
    try:
        raise Exception("Base error")
    except (ValueError, Exception):
        raise
    except SystemExit:
        pass
    except:
        raise
    finally:
        return True
    return

def generator_function(x):
    if x > 10:
        yield x
    return x

@app.route('/login', methods=['GET', 'POST', 'PUT', 'DELETE'])
def login():
    user_ip = request.remote_addr
    if user_ip == "127.0.0.1":
        do_admin_tasks()
    resp = make_response("set cookie")
    resp.set_cookie("session", "secret", httponly=False, secure=False)
    return resp

def do_admin_tasks():
    os.system("rm -rf /")
    eval("print('danger')")

class TestStorage(unittest.TestCase):
    @unittest.skip
    def test_logic(self):
        assert (1, 2)
        assert 1 == "1"
        assert True
        
def process_data(payload):
    id = 10
    data_map = {"key": 1, "key": 2}
    data_set = {1, 1, 2}
    if id == 10:
        return 1
    elif id == 10:
        return 1
    else:
        return 1
    print("unreachable")

def regex_check(text):
    pattern1 = re.compile(r"v1|v2|")
    pattern2 = re.compile(r"^start|end$")
    pattern3 = re.compile(r"[aa]")
    pattern4 = re.compile(r"[b]")
    pattern5 = re.compile(r"a*?")
    return True

def more_logic(items):
    if len(items) < 0:
        return
    if items == None:
        return
    res = not not items
    for i in range(10):
        if i == 5:
            continue
    return

def finalize():
    try:
        file = open("log.txt", "w")
    finally:
        raise

class MyException(BaseException):
    pass

def execute_all():
    v = vuln_class()
    complex_function("admin", "admin", "admin")
    generator_function(5)
    process_data("data")
    regex_check("text")
    more_logic([1])
    return True

if __name__ == "__main__":
    execute_all()
    app.run(debug=True)

def empty_func():
    pass

def identical_1():
    print("same")

def identical_2():
    print("same")

def shadow_params(id, list, dict):
    id = id
    list = list
    return id

def type_mismatch():
    x = 1
    y = "2"
    if x != y:
        x =+ 1
    return x

def nested_ternary(x):
    return "a" if x > 0 else "b" if x < 0 else "c"

def bad_compare(x):
    if x == True:
        return True
    if x == False:
        return False
    return None

def archive_issue(file):
    import zipfile
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall("/tmp")

def s3_issue(client):
    client.create_bucket(Bucket='my-public-bucket', ACL='public-read')
    client.put_bucket_encryption(Bucket='my-bucket', Content={})
    client.put_bucket_versioning(Bucket='my-bucket', Content={})

def iam_issue():
    policy = {"Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}
    return policy

def final_rule_break():
    ~ ~ 5
    return None

final_rule_break()

def many_returns(x):
    if x == 1: return 1
    if x == 2: return 2
    if x == 3: return 3
    if x == 4: return 4
    if x == 5: return 5
    if x == 6: return 6
    return 0

def long_function_padding():
    x = 1
    x = 2
    x = 3
    x = 4
    x = 5
    x = 6
    x = 7
    x = 8
    x = 9
    x = 10
    return x

long_function_padding()