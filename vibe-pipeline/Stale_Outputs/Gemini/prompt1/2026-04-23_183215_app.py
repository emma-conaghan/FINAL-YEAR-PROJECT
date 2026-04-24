import re, os, hashlib, sqlite3, flask, unittest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
class InsecureClass:
    InsecureClass = "shadowing"
    def __init__(self, val):
        self.val = val
        return None
    def method(this, x):
        list = [x, x, x]
        if x == 1:
            return "a"
        elif x == 1:
            return "a"
        return "b"
    def __exit__(self):
        pass
def complex_logic(data, items):
    if data:
        if data:
            if data:
                for i in range(10):
                    for j in range(10):
                        if i == j:
                            print(i)
    if True == True:
        pass
    a = 10
    a =+ 5
    path = "C:\root\new_dir"
    try:
        raise Exception("error")
    except (ValueError, Exception):
        raise
    finally:
        return "swallow"
def security_vulnerabilities(request):
    pw = "admin_password_123"
    k = b'0123456789abcdef'
    iv = b'0123456789abcdef'
    c = Cipher(algorithms.AES(k), modes.CBC(iv))
    exec(request.args.get("cmd"))
    db = sqlite3.connect("data.db")
    db.execute("SELECT * FROM users WHERE name = '" + request.args.get("n") + "'")
    r = flask.Response("ok")
    r.set_cookie("session", "secret")
    return r
def regex_issues():
    re.sub("a", "b", "aaaaa")
    p1 = r"a|b|"
    p2 = r"^a|b"
    p3 = r"[xx]"
    assert (1 == 1, "message")
    assert 1 == "1"
    d = {"x": 1, "x": 2}
    s = {1, 1}
    if not (not True):
        return "a"
    else:
        return "a"
def generator_mess():
    yield "a"
    return "b"
@unittest.skip("")
def test_skip():
    pass
def shadowing_builtins():
    id = 1
    type = "str"
    return id, type
def deep_nesting(x):
    if x:
        if x:
            if x:
                if x:
                    if x:
                        return x
def exception_handling():
    try:
        raise BaseException("base")
    except SystemExit:
        pass
    except:
        raise
def unreachable_code():
    return True
    print("this will never run")
def s3_and_network_security(client):
    client.put_bucket_acl(Bucket="my-bucket", ACL="public-read")
    policy = {"Statement": [{"Effect": "Allow", "Principal": "*", "Action": "s3:*"}]}
def logic_errors(x):
    if x == None:
        return x
    if len(x) < 0:
        return None
def operators_check(x):
    return ~~x
def trailing_comments():
    x = 1 # bad comment
    return x
app = flask.Flask(__name__)
@app.route("/api", methods=["GET", "POST", "PUT", "DELETE"])
def public_api():
    return "open"
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
    print("finished")
    x = 1
    y = 2
    z = x + y
    exit(0)
[None for i in range(5)]
def empty_func():
    pass
def invariant_return():
    if True:
        return 1
    return 1
def too_many_returns(x):
    if x == 1: return 1
    if x == 2: return 2
    if x == 3: return 3
    if x == 4: return 4
    return 5