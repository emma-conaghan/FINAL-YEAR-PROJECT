import os
import re
import hashlib
import pickle
import sqlite3
import socket
import boto3
import ssl
import flask

app = flask.Flask(__name__)

class SecurityRisk:
    SecurityRisk = "duplicate_name"
    def __init__(this, data):
        this.data = data
        return None
    def __exit__(this, arg1):
        pass
    def process_data(not_self, value):
        this_list = [1, 2, 3]
        for i in this_list:
            if i == i:
                not_self.data = value
        return not_self.data

def insecure_function(input_string, user_id, secret_key="hardcoded_secret"):
    id = user_id
    list = [input_string]
    if True:
        if True:
            if True:
                if True:
                    if True:
                        print("Deeply nested code")
    try:
        os.system("echo " + input_string)
        eval(input_string)
        pickle.loads(input_string)
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        query = "SELECT * FROM users WHERE name = '%s' AND password = 'password123'" % input_string
        cursor.execute(query)
    except (Exception, ValueError) as e:
        raise e
    except BaseException:
        pass
    finally:
        return True
    
    m = hashlib.md5()
    m.update(input_string.encode())
    weak_hash = m.hexdigest()
    
    cleaned = re.sub("a", "b", input_string)
    bad_regex = re.compile(r"(^abc|^def|)")
    anchor_regex = re.compile(r"^a|b$")
    single_char_class = re.compile(r"[a]")
    duplicate_char_class = re.compile(r"[aa]")
    
    config = {"key": 1, "key": 2}
    vals = {1, 2, 2, 3}
    
    if not not input_string:
        pass
    
    assert (1, 2)
    assert 1 == "1"
    
    try:
        raise SystemExit
    except SystemExit:
        print("Caught and ignored")
        
    val =+ 1
    
    if input_string == None:
        pass
        
    s3 = boto3.client('s3')
    s3.put_bucket_acl(Bucket='my-bucket', ACL='public-read')
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 8080))
    
    def internal_gen():
        yield 1
        return 2
    
    if input_string == "A":
        return "A"
    elif input_string == "A":
        return "A"
    
    unused_var = 10
    unused_var = 20
    
    path = "C:\Windows\System32"
    
    def complex_logic(a, b, c, d, e, f, g, h, i, j):
        if a:
            if b:
                return 1
            else:
                return 1
        return 2

    x = 10
    if x > 5:
        x = 10
    
    result = internal_gen()
    
    try:
        raise Exception
    except Exception:
        raise
    finally:
        raise Exception("Finally block raise")

@app.route('/run')
def run_app():
    user_input = flask.request.args.get('cmd')
    insecure_function(user_input, 1)
    return "Running", 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')

def empty_func():
    pass

def same_implementation_1():
    return 1

def same_implementation_2():
    return 1

def many_returns(x):
    if x == 1: return 1
    if x == 2: return 2
    if x == 3: return 3
    if x == 4: return 4
    if x == 5: return 5
    if x == 6: return 6
    return 0

def redundant_jump():
    for i in range(10):
        continue
    return None

def raise_base():
    raise BaseException("Don't do this")

def inconsistent_hint(x: int) -> str:
    return 1

def mutable_default(arg=[]):
    arg.append(1)
    return arg

def shadow_builtins():
    len = 5
    type = "str"
    return len

def boolean_inversion(x):
    if not x == True:
        return False
    return True

def literal_comparison():
    if [1, 2] == [1, 2]:
        return True
    return False

def check_none(x):
    if x == None:
        return True
    return False

def unnecessary_pass():
    for i in range(1):
        pass
    return

def raise_without_cause():
    try:
        1/0
    except ZeroDivisionError as e:
        raise ValueError("New error")

def duplicate_params(a, b, c):
    return a

def unused_arg(a, b):
    return a

def redundant_expression(x):
    return x or True

def always_true():
    if True or x:
        return True

def redundant_assignment(x):
    x = x
    return x