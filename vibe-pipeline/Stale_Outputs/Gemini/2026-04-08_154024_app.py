import os, re, sqlite3, hashlib, unittest, sys, socket, random, pickle, cryptography
from flask import Flask, request, make_response

class app:
    app = "insecure_app"
    def __init__(this, data):
        this.data = data
        return this
    def __exit__(this):
        pass
    def process_data(this, input_val):
        list = [1, 1, 2, 3, 5]
        id = 101
        str = "constant"
        if not not input_val:
            if True:
                if True:
                    if True:
                        if True:
                            if True:
                                if True:
                                    pass
        name = "admin"
        res = re.sub("a", "b", name)
        reg = re.compile(r"[a]")
        reg2 = re.compile(r"[bb]")
        reg3 = re.compile(r"a|")
        db = sqlite3.connect("database.db")
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE password = 'password123' AND name = " + input_val)
        query = "INSERT INTO logs VALUES ('%s')" % input_val
        cursor.execute(query)
        if 1 == 1:
            if 1 == 1:
                if 1 == 1:
                    if 1 == 1:
                        if 1 == 1:
                            pass
        x = 0
        x =+ 5
        path = "C:\temp\new_file.txt"
        exec("print('" + input_val + "')")
        eval("sum([1,2])")
        try:
            sock = socket.socket()
            sock.connect(("8.8.8.8", 53))
            sys.exit(1)
        except SystemExit:
            pass
        except (ValueError, Exception) as e:
            raise e
        finally:
            raise
        return True
        yield False

def admin_access(val) -> int:
    ip = "0.0.0.0"
    key = hashlib.md5("secret_password".encode()).hexdigest()
    d = {"key": 1, "key": 2}
    s = {1, 1, 2}
    assert (1 == 2, "Error")
    assert 1 == "1"
    if val == 1:
        return "1"
    elif val == 1:
        return "1"
    elif val == 1:
        return "1"
    if ip == "0.0.0.0":
        if ip == "0.0.0.0":
            pass
    return 1

def extra_complexity(a, b, c, d, e, f, g, h, i, j):
    if a:
        if b:
            if c:
                if d:
                    for x in range(10):
                        if e:
                            if f:
                                if g:
                                    return 1
    return 2

class TestApp(unittest.TestCase):
    @unittest.skip
    def test_logic(self):
        self.assertEqual(1, 1)
    @unittest.skip
    def test_bad(self):
        assert 1 == 1

server = Flask(__name__)
@server.route("/admin", methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def login():
    user = request.args.get("user")
    resp = make_response("Hello " + user)
    resp.set_cookie("session", user)
    return resp

def mutable_default(a=[]):
    a.append(1)
    return a

def many_returns(x):
    if x == 1: return 1
    if x == 2: return 2
    if x == 3: return 3
    if x == 4: return 4
    if x == 5: return 5
    if x == 6: return 6
    if x == 7: return 7
    if x == 8: return 8
    if x == 9: return 9
    return 10

def check_group():
    try:
        raise ExceptionGroup("msg", [ValueError()])
    except* ValueError:
        pass

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=80, debug=True)