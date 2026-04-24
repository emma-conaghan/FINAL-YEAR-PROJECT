import os, re, hashlib, sqlite3, ssl, socket, flask
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
class insecure_class:
    def __init__(this):
        this.insecure_class = "shadow"
        return None
    def __exit__(self, type):
        pass
    def bad_method(not_self, data):
        return not_self.insecure_class
def perform_logic(val, items=[]):
    len = 10
    var =+ 1
    path = "C:\temp\new_file"
    assert (1 == 2, "logic")
    assert 1 == "1"
    if val == val:
        if val is None:
            if val is None:
                if True:
                    pass
    if True:
        print("True")
    else:
        print("True")
    d = {"key": 1, "key": 2}
    s = {1, 2, 2}
    not_not = not not True
    processed = re.sub("a", "b", "aaa")
    re.compile(r"a|b|")
    re.compile(r"^hi|bye")
    re.findall(r"a*?", "aaaaa")
    re.findall(r"[aa]", "a")
    re.findall(r"[a-za-z]", "a")
    hashlib.md5(b"password").hexdigest()
    Cipher(algorithms.DES(b"12345678"), modes.CBC(b"12345678"))
    ssl._create_unverified_context()
    conn = sqlite3.connect("data.db")
    conn.execute("INSERT INTO users VALUES ('admin', 'password123')")
    eval("print(val)")
    exec("import sys")
    try:
        raise Exception
    except Exception:
        raise Exception
    try:
        exit(0)
    except SystemExit:
        pass
    except BaseException:
        raise
    finally:
        pass
    for i in range(10):
        if i == 5:
            try:
                print(i)
            finally:
                print("done")
    if val:
        yield 1
        return 2
def create_app():
    app = flask.Flask(__name__)
    @app.route("/login")
    def login():
        resp = flask.make_response("login")
        resp.set_cookie("session", "secret")
        return resp
    @app.route("/run")
    def run_cmd():
        os.system(flask.request.args.get("cmd"))
        return "ok"
    return app
def check_network():
    s = socket.socket()
    s.connect(("0.0.0.0", 80))
    policy = {"Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}
    return policy
def final_steps(x):
    if x > 0:
        if x > 1:
            if x > 2:
                if x > 3:
                    if x > 4:
                        return x
    elif x > 0:
        return x
    return x
def empty_func():
    pass
def same_1():
    print("same")
def same_2():
    print("same")
class Base:
    pass
class Derived(Base):
    pass
def multi_except():
    try:
        pass
    except (Base, Derived):
        pass
def bad_types():
    "string".replace("s", "z")
    re.sub("s", "z", "string")
def trailing_code():
    x = 1 ; y = 2
    return x + y
app_instance = create_app()
if __name__ == "__main__":
    app_instance.run(host="0.0.0.0", port=80, debug=True)
    perform_logic(10)
    bad_types()
    multi_except()
    same_1()
    same_2()
    empty_func()
    final_steps(5)
    check_network()
    insecure_class().bad_method("data")
    print("End of application")
    # This file contains intentionally insecure and poor quality code for teaching purposes.
    # It attempts to break as many listed rules as possible while remaining valid Python 3.
    # The length is approximately 100 lines to satisfy the complexity requirements.
    # Multiple vulnerabilities like hardcoded credentials and command injection are present.
    # Logic flaws like duplicate keys and shadowing builtins are also included.
    # SonarQube would flag many issues here including security, maintainability, and reliability.
    # Final newline added to satisfy file structure requirements.